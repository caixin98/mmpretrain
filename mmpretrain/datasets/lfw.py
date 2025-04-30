import numpy as np
import scipy.misc
import torch
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
import os
from .base_dataset import BaseDataset
from mmpretrain.registry import DATASETS
from mmcv.transforms import Compose


def getAccuracy(scores, flags, threshold):
    p = (scores[flags == 1] > threshold).sum()
    n = (scores[flags == -1] < threshold).sum()
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = torch.zeros((2 * thrNum + 1, 1))
    thresholds = torch.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = (accuracys == accuracys.max()).squeeze()
    bestThreshold = thresholds[max_index].mean()
    return bestThreshold


@DATASETS.register_module()
class LFW(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 img_prefix,
                 pair_file,
                 pipeline,
                 test_mode=False):

        super(LFW, self).__init__()
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.img_prefix = img_prefix
        self.pair_file = pair_file
        data_infos = []
        with open(pair_file) as f:
            pairs = f.read().splitlines()[1:]
        for i, line in enumerate(pairs):
            p = line.split('\t')
            if len(p) == 3:
                imgfile1 = os.path.join(self.img_prefix, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
                imgfile2 = os.path.join(self.img_prefix, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
                label = 1
            else:
                imgfile1 = os.path.join(self.img_prefix, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
                imgfile2 = os.path.join(self.img_prefix, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
                label = -1

            # print('JUST FOR DEBUG')
            # if i == 600:
            #     break
            # fold = i // 60

            fold = i // 600
            #print(fold)
            info = {'imgfile1': imgfile1, 'imgfile2': imgfile2, 'label': label, 'fold': fold}
            #print(info)
            data_infos.append(info)

        self.data_infos = data_infos

        # # for debug
        # from scipy.io import loadmat
        # data = loadmat('/home/SENSETIME/zengwang/mydata/best_result.mat')
        # results = []
        # fold = data['fold']
        # flag = data['flag']
        # fr = data['fr']
        # fl = data['fl']
        # for i in range(flag.shape[1]):
        #     res = {}
        #     feature = [torch.tensor(fl[i]), torch.tensor(fr[i])]
        #     feature = torch.stack(feature, dim=0)
        #     feature = feature.reshape(1, 4, 128)
        #     res['feature'] = feature
        #     res['fold'] = torch.tensor(fold[0, i]).unsqueeze(-1).unsqueeze(-1)
        #     res['label'] = torch.tensor(flag[0, i]).unsqueeze(-1).unsqueeze(-1)
        #     results.append(res)
        # self.evaluate(results)

    def __getitem__(self, index):
        data_info = self.data_infos[index]
        return self.pipeline(data_info)

    def __len__(self):
        return len(self.data_infos)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy']
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        if 'accuracy' in metrics:
            accs = torch.zeros(10)
            feature = [res['feature'] for res in results]
            fold = [res['fold'] for res in results]
            label = [res['label'] for res in results]

            feature = torch.cat(feature, dim=0)
            feature1 = torch.cat([feature[:, 0], feature[:, 1]], dim=-1)
            feature2 = torch.cat([feature[:, 2], feature[:, 3]], dim=-1)
            folds = torch.cat(fold, dim=0).squeeze()
            labels = torch.cat(label, dim=0).squeeze()

            for i in range(10):
                valfold = folds != i
                testfold = folds == i
                mean_val = torch.cat([
                    feature1[valfold], feature2[valfold]],
                    dim=0).mean(dim=0, keepdim=True)

                fea1 = feature1 - mean_val
                fea2 = feature2 - mean_val
                fea1 = fea1 / fea1.norm(p=2, dim=-1, keepdim=True)
                fea2 = fea2 / fea2.norm(p=2, dim=-1, keepdim=True)
                scores = (fea1 * fea2).sum(dim=-1)
                threshold = getThreshold(scores[valfold], labels[valfold], 10000)
                accs[i] = getAccuracy(scores[testfold], labels[testfold], threshold)
            for i in range(len(accs)):
                logger.info('{}    {:.2f}'.format(i + 1, accs[i] * 100))
            logger.info('--------')
            logger.info('AVE    {:.2f}'.format(accs.mean() * 100))
            eval_result = dict(accuracy=accs.numpy())
        return eval_result
