from torch import nn
import torch.nn.functional as F
import torch
import warnings
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale, resize
import numpy as np
from waveprop.slm import get_active_pixel_dim, get_slm_mask
from waveprop.pytorch_util import fftconvolve
from waveprop.spherical import spherical_prop
from waveprop.color import ColorSystem
from waveprop.rs import angular_spectrum
from lensless.constants import RPI_HQ_CAMERA_BLACK_LEVEL
from skimage.util.noise import random_noise
from scipy import ndimage
from lenslessclass.util import AddPoissonNoise
from waveprop.devices import SensorParam
from collections import OrderedDict
import cv2
from mmpretrain.registry import MODELS

@MODELS.register_module()
class SlmPsfConv(nn.Module):
    """ 
    given slm_config, sensor_config, crop_fact, scene2mask, mask2sensor,
    do the convolution with slm psf
    and training the parameters of slm (mask)
    """

    def __init__(
        self,
        input_shape,
        slm_config,
        sensor_config,
        crop_fact,
        scene2mask,
        mask2sensor,
        target_dim=None,  # try to get this while keeping aspect ratio of sensor
        down="resize",
        device="cpu",
        dtype=torch.float32,
        deadspace=True,
        first_color=0,
        grayscale=True,
        device_mask_creation=None,
        multi_gpu=False,  # or pass list of devices
        sensor_activation=None,
        dropout=None,
        noise_type=None,
        snr=40,
        return_measurement=False,
        requires_grad = True,
        output_dim=None,
        n_slm_mask=1,  # number of SLM masks to optimize, would be time-multiplexed
        **kwargs,
    ):
        """
        grayscale : whether input is grayscale, can then simplify to just grayscale PSF

        """
        super(SlmPsfConv, self).__init__()

        if dtype == torch.float32:
            self.ctype = torch.complex64
        elif dtype == torch.float64:
            self.ctype = torch.complex128
        else:
            raise ValueError(f"Unsupported data type : {dtype}")

        if len(input_shape) == 2:
            input_shape = [1] + list(input_shape)
        assert len(input_shape) == 3

        # store configuration
        self.input_shape = np.array(input_shape)
        
        self.slm_config = slm_config
        self.sensor_config = sensor_config
        self.crop_fact = crop_fact
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.deadspace = deadspace
        self.first_color = first_color
        self.grayscale = grayscale
        self.device = device
        if device_mask_creation is None:
            device_mask_creation = device
        else:
            self.device_mask_creation = device_mask_creation
        self.dtype = dtype
        self.target_dim = target_dim
        self.return_measurement = return_measurement
        self.requires_grad = requires_grad  # for SLM vals
        self.n_slm_mask = n_slm_mask

        # adding noise
        if noise_type:
            if noise_type == "poisson":
                add_noise = AddPoissonNoise(snr)

            else:
                # TODO : hardcoded for Raspberry Pi HQ sensor
                bit_depth = 12
                noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)

                def add_noise(measurement):

                    # normalize as mean is normalized to max value 1
                    with torch.no_grad():
                        max_vals = torch.max(torch.flatten(measurement, start_dim=1), dim=1)[0]
                        max_vals = max_vals.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    measurement /= max_vals

                    # compute noise for each image
                    measurement_np = measurement.clone().cpu().detach().numpy()
                    noise = []
                    for _measurement in measurement_np:

                        sig_var = ndimage.variance(_measurement)
                        noise_var = sig_var / (10 ** (snr / 10))
                        noise.append(
                            random_noise(
                                _measurement,
                                mode=noise_type,
                                clip=False,
                                mean=noise_mean,
                                var=noise_var,
                            )
                            - _measurement
                        )

                    noise = torch.tensor(np.array(noise).astype(np.float32)).to(device)

                    return measurement + noise

            self.add_noise = add_noise
        else:
            self.add_noise = None

        # -- downsampling
        self.downsample = None

        if self.target_dim is not None or output_dim is not None:
            if output_dim is None:
                sensor_size = sensor_config[SensorParam.SHAPE]
                w = np.sqrt(np.prod(self.target_dim) * sensor_size[1] / sensor_size[0])
                h = sensor_size[0] / sensor_size[1] * w
                self.output_dim = np.array([int(h), int(w)])
            else:
                self.output_dim = np.array(output_dim)

            if not grayscale:
                self.output_dim = np.r_[self.output_dim, 3]

            if down == "resize":

                self.downsample = transforms.Resize(size=self.output_dim[:2].tolist())

            elif down == "max" or down == "avg":
                n_embedding = np.prod(self.output_dim)

                # determine filter size, stride, and padding: https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/
                k = int(np.ceil(np.sqrt(np.prod(self.input_shape[1:]) / n_embedding)))
                p = np.roots(
                    [
                        4,
                        2 * np.sum(self.input_shape),
                        np.prod(self.input_shape[1:]) - k**2 * n_embedding,
                    ]
                )
                p = max(int(np.max(p)), 0) + 1
                if down == "max":
                    self.downsample = nn.MaxPool2d(kernel_size=k, stride=k, padding=p)
                else:
                    self.downsample = nn.AvgPool2d(kernel_size=k, stride=k, padding=p)
                pooling_outdim = ((self.input_shape[1:] - k + 2 * p) / k + 1).astype(int)
                assert np.array_equal(self.output_dim[:2], pooling_outdim)
            else:
                raise ValueError("Invalid downsampling approach.")
        else:
            self.output_dim = self.input_shape


        # -- decision network after sensor
        self.sensor_activation = sensor_activation
        self.flatten = nn.Flatten()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # -- determine number of active SLM pixels
        overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
            sensor_config=sensor_config,
            sensor_crop=crop_fact,
            slm_config=slm_config,
        )

        # -- initialize SLM values, set as parameter to optimize
        # TODO : need to make internal changes to get_slm_mask for no deadspace
        rand_func = torch.rand  # between [0, 1]
        # rand_func = torch.randn
        if deadspace:
            if self.n_slm_mask == 1:
                self.slm_vals = rand_func(
                    *n_active_slm_pixels,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )
                if self.requires_grad:
                    self.slm_vals = nn.Parameter(self.slm_vals)
            else:
                self.slm_vals = [
                    rand_func(
                        *n_active_slm_pixels,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=self.requires_grad,
                    )
                    for _ in range(self.n_slm_mask)
                ]
                if self.requires_grad:
                    self.slm_vals = nn.ParameterList(
                        [nn.Parameter(self.slm_vals[i]) for i in range(self.n_slm_mask)]
                    )

        else:
            if self.requires_grad:
                self.slm_vals = nn.Parameter(
                    rand_func(
                        *overlapping_mask_dim,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=self.requires_grad,
                    )
                )
            else:
                self.slm_vals = rand_func(
                    *overlapping_mask_dim,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )

        # -- normalize after PSF
        if self.grayscale:
            self.conv_bn = nn.BatchNorm2d(self.n_slm_mask)
        else:
            self.conv_bn = nn.BatchNorm2d(self.n_slm_mask * 3)

        # -- initialize PSF from SLM values and pre-compute constants
        # object to mask (modeled with spherical propagation which can be pre-computed)
        self.color_system = ColorSystem.rgb()
        self.d1 = np.array(overlapping_mask_size) / self.input_shape[1:]
        self.spherical_wavefront = spherical_prop(
            in_shape=self.input_shape[1:],
            d1=self.d1,
            wv=self.color_system.wv,
            dz=self.scene2mask,
            return_psf=True,
            is_torch=True,
            device=self.device,
            dtype=self.dtype,
        )

        self._psf = None
        self._H = None  # pre-compute free space propagation kernel
        self._H_exp = None

        slm_vals = self.get_slm_vals()
        self.compute_intensity_psf(slm_vals=slm_vals)

        # -- parallelize across GPUs
        if multi_gpu:
            self.downsample = nn.DataParallel(self.downsample, device_ids=multi_gpu)
            if self.classifier is not None:
                self.classifier = nn.DataParallel(self.classifier, device_ids=multi_gpu)
            else:
                self.conv_bn = nn.DataParallel(self.conv_bn, device_ids=multi_gpu)
                self.linear1 = nn.DataParallel(self.linear1, device_ids=multi_gpu)
                if self.linear2:
                    self.linear2 = nn.DataParallel(self.linear2, device_ids=multi_gpu)
                    if self.bn is not None:
                        self.bn = nn.DataParallel(self.bn, device_ids=multi_gpu)

    def get_slm_vals(self):
        """
        Apply any pre-processing to learned SLM values, e.g. clamping within [0, 1].

        TODO quantize / use look up table
        """

        # TRACKING GRADIENT WITH CLAMPING
        if self.n_slm_mask == 1:
            # slm_vals = self.slm_vals.sigmoid()
            slm_vals = self.slm_vals.clamp(min=0, max=1)  # found clamp to be better
        else:
            # slm_vals = [self.slm_vals[i].sigmoid() for i in range(self.n_slm_mask)]
            slm_vals = [self.slm_vals[i].clamp(min=0, max=1) for i in range(self.n_slm_mask)]

        return slm_vals

    def set_slm_vals(self, slm_vals):
        """
        only works if requires_grad = False
        """

        np.testing.assert_array_equal(slm_vals.shape, self.slm_vals.shape)
        self.slm_vals = slm_vals

        # recompute intensity PSF
        self.compute_intensity_psf()

    def get_psf(self, numpy=False):
        slm_vals = self.get_slm_vals()
        self.compute_intensity_psf(slm_vals=slm_vals)

        if numpy:
            if self.n_slm_mask == 1:
                return self._psf.cpu().detach().numpy().squeeze()
            else:
                return [_psf.cpu().detach().numpy().squeeze() for _psf in self._psf]
        else:
            return self._psf

    def set_mask2sensor(self, mask2sensor):
        self.mask2sensor = mask2sensor

        # recompute intensity PSF
        self.compute_intensity_psf()

    def save_psf(self, fp, bit_depth=8):
        psf = self.get_psf(numpy=True)
        psf = np.transpose(psf, (1, 2, 0))
        psf /= psf.max()

        # save as int
        psf *= 2**bit_depth - 1
        if bit_depth <= 8:
            psf = psf.astype(dtype=np.uint8)
        else:
            psf = psf.astype(dtype=np.uint16)
        cv2.imwrite(fp, cv2.cvtColor(psf, cv2.COLOR_RGB2BGR))

    def forward(self, x):

        if x.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            x -= x.min()

        # compute intensity PSF from SLM values
        slm_vals = self.get_slm_vals()  # apply any (physical) pre-processing
        self.compute_intensity_psf(slm_vals=slm_vals)

        if self.n_slm_mask > 1:
            # add dimension for time-multiplexed measurements
            x = x.unsqueeze(1)

        # convolve with PSF
        x = fftconvolve(x, self._psf, axes=(-2, -1))

        if self.n_slm_mask > 1:
            # consider time-multiplexed as more channels
            x = x.flatten(1, 2)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.add_noise is not None:
            x = self.add_noise(x)

        # make sure non-negative
        x = torch.clip(x, min=0)

        if self.return_measurement:
            return x

        # normalize after PSF
        x = self.conv_bn(x)

        if self.sensor_activation is not None:
            x = self.sensor_activation(x)

   
        return x

    def compute_intensity_psf(self, slm_vals=None):

        if slm_vals is None:
            slm_vals = self.get_slm_vals()

        assert slm_vals.max() <= 1
        assert slm_vals.min() >= 0

        if self.n_slm_mask == 1:
            # TODO : backward compatability but can make consistent with multiple masks

            # -- get SLM mask, i.e. deadspace modeling, quantization (todo), non-linearities (todo), etc
            mask = get_slm_mask(
                slm_vals=slm_vals.to(self.device_mask_creation),
                slm_config=self.slm_config,
                sensor_config=self.sensor_config,
                crop_fact=self.crop_fact,
                target_dim=self.input_shape[1:],
                deadspace=self.deadspace,
                device=self.device_mask_creation,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )
            if self.device != self.device_mask_creation:
                mask = mask.to(self.device)

            # apply mask
            u_in = mask * self.spherical_wavefront

            # mask to sensor
            if self._H is None:
                # precompute at very start, not dependent on input pattern, just its shape
                # TODO : benchmark how much it actually saves
                self._H = torch.zeros(
                    [self.color_system.n_wavelength] + list(self.input_shape[1:] * 2),
                    dtype=self.ctype,
                    device=self.device,
                )
                for i in range(self.color_system.n_wavelength):
                    self._H[i] = angular_spectrum(
                        u_in=u_in[i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=self.device,
                        return_H=True,
                    )

            psfs = torch.zeros(u_in.shape, dtype=self.ctype, device=self.device)
            for i in range(self.color_system.n_wavelength):
                psfs[i], _, _ = angular_spectrum(
                    u_in=u_in[i],
                    wv=self.color_system.wv[i],
                    d1=self.d1,
                    dz=self.mask2sensor,
                    dtype=self.dtype,
                    device=self.device,
                    H=self._H[i],
                    # H_exp=self._H_exp[i],
                )

            psfs = psfs / torch.norm(psfs.flatten())

        else:

            assert len(slm_vals) == self.n_slm_mask

            # compute masks
            masks_dim = [self.n_slm_mask, self.color_system.n_wavelength] + list(
                self.input_shape[1:]
            )
            masks = torch.zeros(masks_dim, dtype=self.dtype, device=self.device)
            for i in range(self.n_slm_mask):
                masks[i] = get_slm_mask(
                    slm_vals=slm_vals[i].to(self.device_mask_creation),
                    slm_config=self.slm_config,
                    sensor_config=self.sensor_config,
                    crop_fact=self.crop_fact,
                    target_dim=self.input_shape[1:],
                    deadspace=self.deadspace,
                    device=self.device_mask_creation,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                )

            # apply mask
            u_in = masks * self.spherical_wavefront

            # mask to sensor
            if self._H is None:
                # precompute at very start, not dependent on input pattern, just its shape
                # TODO : benchmark how much it actually saves
                self._H = torch.zeros(
                    [self.color_system.n_wavelength] + list(self.input_shape[1:] * 2),
                    dtype=self.ctype,
                    device=self.device,
                )
                for i in range(self.color_system.n_wavelength):
                    self._H[i] = angular_spectrum(
                        u_in=u_in[0][i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=self.device,
                        return_H=True,
                    )

            psfs = torch.zeros(u_in.shape, dtype=self.ctype, device=self.device)
            for n in range(self.n_slm_mask):
                for i in range(self.color_system.n_wavelength):
                    psfs[n][i], _, _ = angular_spectrum(
                        u_in=u_in[n][i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=self.device,
                        H=self._H[i],
                    )

            norm_fact = (
                torch.norm(psfs.flatten(1, 3), dim=1, keepdim=True).unsqueeze(2).unsqueeze(2)
            )
            psfs = psfs / norm_fact

        # intensity psf
        if self.grayscale:
            self._psf = rgb_to_grayscale(torch.square(torch.abs(psfs)))
        else:
            self._psf = torch.square(torch.abs(psfs))