
from mmpretrain.registry import TRANSFORMS
import mmengine.fileio as fileio
import mmcv
import numpy as np
import torch
import os.path as osp
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LoadImagePair(BaseTransform):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def transform(self, results):
        if self.file_client is None:
            self.file_client = fileio.FileClient(**self.file_client_args)

        imgfile1 = results['imgfile1']
        imgfile2 = results['imgfile2']

        img_bytes = self.file_client.get(imgfile1)
        img1 = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img1 = img1.astype(np.float32)

        img_bytes = self.file_client.get(imgfile2)
        img2 = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img2 = img2.astype(np.float32)

        results['img1'] = img1
        results['img2'] = img2
        results['img_fields'] = ['img1', 'img2']
        return results
    
@TRANSFORMS.register_module()    
class FlipPair(BaseTransform):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility and flip direction.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, keys=['img1', 'img2'], keys_flip=['img1_flip', 'img2_flip'], direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        self.direction = direction
        self.keys = keys
        self.keys_flip = keys_flip

    def transform(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        for key, key_flip in zip(self.keys, self.keys_flip):
            results[key_flip] = mmcv.imflip(
                results[key], direction=self.direction)
            results['img_fields'].append(key_flip)
        return results

@TRANSFORMS.register_module()
class StackImagePair(BaseTransform):
    def __init__(self, keys, out_key, dim=0):
        self.keys = keys
        self.out_key = out_key
        self.dim = dim

    def transform(self, results):
        out = [results[key] for key in self.keys]
        results[self.out_key] = torch.stack(out, dim=0)
        return results
@TRANSFORMS.register_module()
class LoadImageFromFile2(BaseTransform):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def transform(self, results):
        if self.file_client is None:
            self.file_client = fileio.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        # try:
        #     img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        # except:
        #     with open('debug_{}.txt'.format(dist.get_rank()), 'a') as f:
        #         f.write(filename+'\n')
        #     print(filename)
        #     img = np.zeros([400, 400, 3]).astype(np.float32)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
