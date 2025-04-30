# amplitude mask for training

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Function
from mmpretrain.registry import MODELS
from mmpretrain.models.opticals import PsfConv
from mmpretrain.devices import SensorParam
import numpy as np
import warnings
from waveprop.pytorch_util import fftconvolve
from torchvision.transforms.functional import rgb_to_grayscale, center_crop

class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, x):
        x = (x + 1.0) / 2.0
        x = x.clamp(min=0, max=1)
        x = torch.round(x)
        # x = x.clamp(min=0, max=1)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x
    
@MODELS.register_module()   
class SteerPsfConv(PsfConv):
    def __init__(self, feature_size = 3e-5, mask_kernel_shape = None, angle = (0,0),  **kwargs):
        self.binarize = BinarizeFunction.apply
        self.feature_size = feature_size
        self.mask_kernel_shape = mask_kernel_shape
        if isinstance(angle, int):
            self.angle_range = (-angle, angle)
        else:
            self.angle_range = angle
        super().__init__(**kwargs)

    def init_psf_vals(self):
        rand_func = torch.rand
        self.feature_size = np.array(self.feature_size)
        if self.feature_size.shape == 1:
            self.feature_size = np.array([self.feature_size, self.feature_size])

        val_shape = self.sensor_config[SensorParam.SHAPE] * self.sensor_config[SensorParam.PIXEL_SIZE] / self.feature_size 
        val_shape = np.array(val_shape).astype(np.int) // self.split_psf * self.split_psf
        val_shape_split =  val_shape // self.split_psf
        self.val_shape_split = val_shape_split
        self.val_shape = val_shape
        if self.mask_kernel_shape is not None:
            # padding to make sure the size of mask is the same as the size of sensor
            # mask_kernel_shape is like 11,15,63,121 (smaller than the size of sensor_split)
            if isinstance(self.mask_kernel_shape, int):
                self.mask_kernel_shape = [self.mask_kernel_shape]
            padding_masks = [] 
            for mask_kernel_shape in self.mask_kernel_shape:
                mask_kernel_shapes = [mask_kernel_shape,mask_kernel_shape]
                # split padding_mask with self.split
             
                padding_mask = np.zeros(val_shape)
                for i in range(self.split_psf[0]):
                    for j in range(self.split_psf[1]):
                        if val_shape_split[0] < mask_kernel_shapes[0] or val_shape_split[1] < mask_kernel_shapes[1]:
                            left = np.array([i,j]) * val_shape_split
                            right = left + val_shape_split
                        else:
                            left = (val_shape_split - mask_kernel_shape) // 2 + np.array([i,j]) * val_shape_split
                            right = left + mask_kernel_shapes
                        padding_mask[left[0]:right[0], left[1]:right[1]] = 1
                padding_masks.append(padding_mask)
            # padding_mask[left[0]+2, left[1]+2] = 0
            # padding_mask[left[0]+1, left[1]+2] = 0
            # padding_mask[left[0]+2, left[1]+0] = 0
        else:
            padding_masks = [np.ones(val_shape)]
        padding_masks = torch.concat([torch.from_numpy(padding_mask).float().cuda().unsqueeze(0) for padding_mask in padding_masks], dim=0)

        num_mask = padding_masks.shape[0]
          
        if self.n_psf_mask == 1:
            self.psf_vals = rand_func(num_mask, val_shape_split[0], val_shape_split[1], dtype=self.dtype, 
                requires_grad=self.requires_grad, device="cuda")
            # self.psf_vals = psf_vals
            if self.requires_grad:
                self.psf_vals = nn.Parameter(self.psf_vals)

                # self.psf_vals.retain_grad()
        else:
            self.psf_vals = [
                rand_func(num_mask, val_shape[0], val_shape[1], dtype=self.dtype,
                requires_grad=self.requires_grad,device="cuda")
            for _ in range(self.n_psf_mask)]
            # for psf_vals in self.psf_vals:
            #     psf_vals = psf_vals * padding_masks
            #     self.psf_vals.append(psf_vals)
            if self.requires_grad:
                self.psf_vals = [nn.Parameter(psf_vals) for psf_vals in self.psf_vals]
            
                # for psf_vals in self.psf_vals:
                #     psf_vals.retain_grad()
        self.padding_masks = padding_masks
        return self.psf_vals

    def generate_affine_matrix_for_rotate(self, angle):
        angle = angle * np.pi / 180
        affine_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                      [torch.sin(angle), torch.cos(angle), 0]])
        affine_matrix = affine_matrix.unsqueeze(0)
        affine_matrix = affine_matrix.cuda()
        return affine_matrix


    def get_mask_size(self):
        # the size of sensor
        return self.sensor_config[SensorParam.SHAPE] * self.sensor_config[SensorParam.PIXEL_SIZE] 
                
    def get_psf_mask(self, psf_vals,
    mask_dim = None):
        # for n_psf_mask == 1, psf_vals is a tensor of shape [num_mask, h, w]
        num_angle = self.split_psf[0] * self.split_psf[1]
        # split angle_range into num_angle with torch
        angles = torch.linspace(self.angle_range[0], self.angle_range[1], num_angle)
        # angles = np.linspace(self.angle_range[0], self.angle_range[1], num_angle)
        # duplicate psf_vals for each angle
        psf_vals_tensor = torch.zeros(psf_vals.shape[0], num_angle, self.val_shape_split[0], self.val_shape_split[1]).cuda()
        psf_vals = psf_vals.unsqueeze(0)
        for i, angle in enumerate(angles):
            for j in range(psf_vals.shape[1]):
                affine_matrix = self.generate_affine_matrix_for_rotate(angle)
                grid = F.affine_grid(affine_matrix, psf_vals.size(),align_corners=True)
                psf_vals_tensor[j,i] = center_crop(F.grid_sample(psf_vals, grid, align_corners=True,padding_mode="zeros"), self.val_shape_split)
        # reshape psf_vals to [num_mask, h * self.split_psf[0], w * self.split_psf[1]]
        psf_vals_tensor_reshape = torch.zeros((psf_vals.shape[1], self.val_shape[0], self.val_shape[1])).cuda()
        for i in range(self.split_psf[0]):
            for j in range(self.split_psf[1]):
                psf_vals_tensor_reshape[:,i*self.val_shape_split[0]:(i+1)*self.val_shape_split[0],j*self.val_shape_split[1]:(j+1)*self.val_shape_split[1]] = psf_vals_tensor[:,i*self.split_psf[1]+j,:,:]
        psf_vals = psf_vals_tensor_reshape
        
        psf_vals = psf_vals * self.padding_masks
        psf_vals = psf_vals.unsqueeze(0)
        if mask_dim is None:
            mask_dim = self.input_shape[1:]

        psf_vals = F.interpolate(psf_vals,
            size=tuple(mask_dim), mode='nearest-exact')
        # psf_vals_detach = psf_vals.clone().detach() 
        psf_vals = psf_vals - psf_vals.min()
        psf_vals = psf_vals  / (psf_vals.max() + 1e-6)
        # psf_vals = F.sigmoid(psf_vals)
        psf_vals = psf_vals.squeeze(0)
        # print(psf_vals.shape)
        # psf_vals = torch.sum(psf_vals, dim=0)
        # print(psf_vals.shape)
        mask = psf_vals.expand(3,psf_vals.shape[1],psf_vals.shape[2])
        # mask = psf_vals.repeat(3, 1).reshape(3,psf_vals.shape[1],psf_vals.shape[2])
        mask = mask.cuda()
        return mask
    
        
    def compute_intensity_psf(self, psf_vals=None):
        if psf_vals is None:
            psf_vals = self.psf_vals
        mask = self.get_psf_mask(psf_vals=psf_vals)
        self.mask = mask
        # psfs = mask / torch.norm(mask.flatten())
        self._psf = mask
        # if self.grayscale:
        #     self._psf = rgb_to_grayscale(torch.square(torch.abs(psfs)))
        # else:
        #     self._psf = torch.square(torch.abs(psfs))

