from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import local_maxima, heatmap_gen_per_cell


def stain2pos():
    img_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/PHASE").glob("*.tif"))
    gfp_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/GFP").glob("*.tif"))
    rfp_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/RFP").glob("*.tif"))
    save_path_gfp = Path("/home/kazuya/dataset/Riken_2type_cell/GFP_pos")
    save_path_rfp = Path("/home/kazuya/dataset/Riken_2type_cell/RFP_pos")
    save_path_gfp.mkdir(parents=True, exist_ok=True)
    save_path_rfp.mkdir(parents=True, exist_ok=True)

    for paths in zip(img_paths, gfp_paths, rfp_paths):
        img = cv2.imread(str(paths[0]))
        gfp = cv2.imread(str(paths[1]))
        gfp_blur = cv2.GaussianBlur(gfp[:, :, 1], (7, 7), 1.5)
        gfp_blur = (gfp_blur - gfp_blur.min()) / (gfp_blur.max() - gfp_blur.min())
        rfp = cv2.imread(str(paths[2]))
        rfp_blur = cv2.GaussianBlur(rfp[:, :, -1], (7, 7), 1.5)
        rfp_blur = (rfp_blur - rfp_blur.min()) / (rfp_blur.max() - rfp_blur.min())

        gfp_pos = local_maxima(gfp_blur, 0.35, 10)
        rfp_pos = local_maxima(rfp_blur, 0.4, 10)

        # plt.imshow(img), plt.show()
        # plt.imshow(gfp_blur), plt.plot(gfp_pos[:, 0], gfp_pos[:, 1], "rx"), plt.show()
        # plt.imshow(rfp_blur), plt.plot(rfp_pos[:, 0], rfp_pos[:, 1], "rx"),plt.show()

        np.savetxt(str(save_path_gfp.joinpath(paths[1].stem + ".txt")), gfp_pos, fmt="%d")
        np.savetxt(str(save_path_rfp.joinpath(paths[2].stem + ".txt")), rfp_pos, fmt="%d")
        print(1)


def heatmap2ch_gen():
    img_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/PHASE").glob("*.tif"))
    gfp_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/GFP_pos").glob("*.txt"))
    rfp_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/RFP_pos").glob("*.txt"))

    save_path_gfp = Path("/home/kazuya/dataset/Riken_2type_cell/GFP_heatmap")
    save_path_gfp.mkdir(parents=True, exist_ok=True)
    save_path_rfp = Path("/home/kazuya/dataset/Riken_2type_cell/RFP_heatmap")
    save_path_rfp.mkdir(parents=True, exist_ok=True)

    for paths in zip(img_paths, gfp_paths, rfp_paths):
        img = cv2.imread(str(paths[0]))
        gfp_pos = np.loadtxt(str(paths[1]))
        rfp_pos = np.loadtxt(str(paths[2]))

        heatmap_g = heatmap_gen_per_cell(img.shape[:2], gfp_pos, 6)
        heatmap_r = heatmap_gen_per_cell(img.shape[:2], rfp_pos, 6)
        # plt.imshow(heatmap_g), plt.show()
        # plt.imshow(heatmap_r), plt.show()
        # plt.imshow(img), plt.show()

        cv2.imwrite(str(save_path_gfp.joinpath(paths[1].stem + ".tif")), (heatmap_g * 255).astype(np.uint8))
        cv2.imwrite(str(save_path_rfp.joinpath(paths[1].stem + ".tif")), (heatmap_r * 255).astype(np.uint8))
        print("finish 1")
    print("finish")


if __name__ == '__main__':
    # stain2pos()
    heatmap2ch_gen()
