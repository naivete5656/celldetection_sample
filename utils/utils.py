import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import cv2
from scipy.ndimage.filters import gaussian_filter

def gather_path(train_paths, mode):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
    return ori_paths


def show_graph(losses, val_losses):
    x = list(range(len(losses)))
    plt.plot(x, losses)
    plt.plot(x, val_losses)
    plt.show()


def local_maxima(img, threshold, dist):
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_rel=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0)
    return data


def heatmap_gen_per_cell(shape, cell_positions, g_size=9):
    heatmap_basis = np.zeros((g_size * 11, g_size * 11))
    half_size = int((g_size * 11) / 2)
    heatmap_basis[half_size + 1, half_size + 1] = 255
    heatmap_basis = gaussian_filter(heatmap_basis, g_size, mode="constant")
    heatmap_basis = (heatmap_basis - heatmap_basis.min()) / (heatmap_basis.max() - heatmap_basis.min())

    results = [np.zeros((shape[0], shape[1]))]
    for y, x in cell_positions:
        img_t = np.zeros((shape[0] + g_size * 11, shape[1] + g_size * 11))  # likelihood map of one cell
        y_min, y_max = int(y), int(y + (g_size * 11))
        x_min, x_max = int(x), int(x + (g_size * 11))
        img_t[x_min:x_max, y_min:y_max] = heatmap_basis
        results.append(img_t[half_size:-half_size, half_size:-half_size])
    results = np.array(results)
    return np.max(results, 0)

