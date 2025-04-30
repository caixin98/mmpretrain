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
class CropRotatePsfConv(PsfConv):
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
            self.psf_vals = rand_func(num_mask, val_shape[0], val_shape[1], dtype=self.dtype, 
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
    
        psf_vals = torch.sum(psf_vals, dim=0)
        mask = psf_vals.repeat(3, 1).reshape(3,psf_vals.shape[0],psf_vals.shape[1])
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

    def crop_affine(self, x, affine_matrix):
        image_shape = x.shape[-2:]
        cropped_shape = self.output_dim[:2]
        angle_range = self.angle_range
        #calculate crop center list
        top_left = cropped_shape // 2
        bottom_right = image_shape - cropped_shape // 2 - cropped_shape % 2
        center_list = []
        num_center = int(angle_range[1] - angle_range[0] + 1)
        for i in range(num_center):
            center_list.append(top_left + (bottom_right - top_left) * i // (num_center - 1))
        #calculate rotate angle from affine_matrix
        angle = torch.atan2(affine_matrix[0,1], affine_matrix[0,0]) * 180 / np.pi

        #calculate crop center index
        center_index = (angle - angle_range[0]) / (angle_range[1] - angle_range[0]) * (num_center - 1)
        center_index = torch.round(center_index).long()
        # do crop 
        center = np.array(center_list[center_index],dtype=np.int)
        x = x[:,center[0] - cropped_shape[0] // 2:center[0] + cropped_shape[0] // 2 + cropped_shape[0] % 2, center[1] - cropped_shape[1] // 2:center[1] + cropped_shape[1] // 2 + cropped_shape[1] % 2]
        # do rotate, create affine_matrix with angle
        affine_matrix = self.generate_affine_matrix_for_rotate(angle)
        x = x.unsqueeze(0)
        grid = F.affine_grid(affine_matrix, x.size(),align_corners=False)
        x = F.grid_sample(x, grid,align_corners=False)
        x = x.squeeze(0)
        return x
        

    def forward(self, x, affine_matrix = None):
        #print("psf input shape ", x.shape)
        if x.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            x -= x.min()
        self.before_optical = x.detach()
        # print(" self.psf_val.grad is not None ", self.psf_vals.grad)
        # compute intensity PSF from psf values
        psf_vals = self.psf_vals  # apply any (physical) pre-processing
        #print("psf vals shape ", psf_vals.shape)
        self.compute_intensity_psf(psf_vals=psf_vals)
  
        #print("psf shape", self._psf.shape)
        if self.n_psf_mask > 1:
            # add dimension for time-multiplexed measurements
            x = x.unsqueeze(1)
            
    
        # convolve with PSF
        if self.do_optical:
            x = fftconvolve(x, self._psf, axes=(-2, -1))
        #print("output shape: ", x.shape)
        if self.n_psf_mask > 1:
            # consider time-multiplexed as more channels
            x = x.flatten(1, 2)

        # if self.use_stn:
        #     x = self.stn(x)
        # # if affine_matrix is not None and use_stn is false, use affine_matrix to warp the image
        # # elif affine_matrix is not None and not self.do_optical:
        # #     grid = F.affine_grid(affine_matrix, x.size(),align_corners=False)
        # #     x = F.grid_sample(x, grid,align_corners=False)
        # elif self.do_affine:
        #     grid = F.affine_grid(affine_matrix, x.size(),align_corners=False)
        #     x = F.grid_sample(x, grid, align_corners=False)


        if self.add_noise is not None:
            x = self.add_noise(x)

  
        x = torch.clip(x, min=0)
        self.after_optical = x.clone()

        # apply translate and its inverse transform
        # assert affine_matrix is not None
        # new tensor to store the cropped image
        x_cropped = torch.zeros(x.shape[0], x.shape[1], self.output_dim[0], self.output_dim[1]).cuda()
        for i in range(x.shape[0]):
            x_cropped[i] = self.crop_affine(x[i], affine_matrix[i])

        self.after_affine = x_cropped.detach()
 
        if self.return_measurement:
            return x_cropped
        # normalize after PSF 
        x = self.normalize(x_cropped)
        # normalize after PSF
        # x = self.conv_bn(x)
        if self.sensor_activation is not None:
            x = self.sensor_activation(x)
        # x = x.reshape(x.shape[0], x.shape[1] * self.split_psf[0] * self.split_psf[1], x.shape[2] // self.split_psf[0], x.shape[3] // self.split_psf[1])

        return x