import numpy as np
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import os 
import cv2 as cv
from torchvision.utils import save_image
@HOOKS.register_module()
class VisualAfterOpticalHook(Hook):
    def __init__(self, visual_freq = 100) -> None:
        super().__init__()
        self.visual_freq = visual_freq
    def binarization(self,x):
        y = (x + 1.0) / 2.0
        y = np.clip(y, 0, 1) 
        y = np.round(y) * 255
        return y
    def normalize_01(self,x):
        mmin = x.min()
        mmax = x.max()
        print(mmin, mmax)
        y = (x-mmin)/(mmax-mmin)
        return y

    def after_train_iter(self, runner):
        if runner.iter % self.visual_freq == 0 and runner.model.device_ids[0] == 0:
            os.makedirs(os.path.join(runner.work_dir,"visualizations"),exist_ok=True)
   
            # conv_weight = runner.model.module.backbone.conv0.weight.cpu().detach().numpy()
            if runner.model.module.backbone.optical.use_stn:
                print(runner.model.module.backbone.optical.stn.theta[:5])
                print(runner.model.module.backbone.optical.stn.xs[:5])
            after_optical = runner.model.module.backbone.optical.after_optical
            before_optical = runner.model.module.backbone.optical.before_optical
            after_affine = runner.model.module.backbone.optical.after_affine
            # for i in range(after_optical.shape[0]):
            after_save_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_after_optical"  + ".png")
            save_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_before_optical"  + ".png")
            after_affine_save_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_after_affine"  + ".png")
            # print(after_optical.shape,before_optical.shape)
            if len(after_optical) >=16:
                after_optical = save_image(after_optical[:16,:,:,:],after_save_path, nrow = 4, normalize = True)
                before_optical = save_image(before_optical[:16,:,:,:],save_path, nrow = 4, normalize = True)
                after_affine = save_image(after_affine[:16,:,:,:],after_affine_save_path, nrow = 4, normalize = True)
            else:
                after_optical = save_image(after_optical[:1,:,:,:],after_save_path, nrow = 1, normalize = True)
                before_optical = save_image(before_optical[:1,:,:,:],save_path, nrow = 1, normalize = True)
            
        
               