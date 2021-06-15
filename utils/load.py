import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate


def local_maxima(img, threshold=100, dist=2):
    assert len(img.shape) == 2
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data


class CellImageLoad(object):
    def __init__(self, ori_path, gfp_gt_path, rfp_gt_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gfp_gt_paths = gfp_gt_path
        self.rfp_gt_paths = rfp_gt_path
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), 0)
        img = img / img.max()

        gt_name = self.gfp_gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gfp = gt / gt.max()

        gt_name = self.rfp_gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        rfp = gt / gt.max()

        gt = np.vstack([gfp[None, :, :], rfp[None, :, :]])

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        gt_new = np.zeros((2, img.shape[0], img.shape[1]))
        gt_new[0] = gt[0][top:bottom, left:right]
        gt_new[1] = gt[1][top:bottom, left:right]

        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt_new[0] = rotate(gt_new[0], 90 * rand_value)
        gt_new[1] = rotate(gt_new[1], 90 * rand_value)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt_new.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt}

        return datas
