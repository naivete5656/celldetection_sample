from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from core import *
from utils import CellImageLoad, gather_path, show_graph, VisdomClass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from networks import UNet
import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="./image/train",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--val_path",
        dest="val_path",
        help="validation data path",
        default="./image/val",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=8, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=500, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )

    args = parser.parse_args()
    return args


def train(args):
    data_loader = CellImageLoad(args.train_img_path, args.train_gfp_path, args.train_rfp_path)
    train_dataset_loader = torch.utils.data.DataLoader(
        data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    number_of_traindata = data_loader.__len__()

    data_loader = CellImageLoad(args.val_img_path, args.val_gfp_path, args.val_rfp_path)
    val_loader = torch.utils.data.DataLoader(
        data_loader, batch_size=5, shuffle=False, num_workers=0
    )

    save_weight_path = args.save_weight_path
    save_weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_weight_path.parent.joinpath("epoch_weight").mkdir(parents=True, exist_ok=True)
    print(
        "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
            args.epochs, args.batch_size, args.learning_rate, args.gpu
        )
    )

    # define model
    net = UNet(n_channels=1, n_classes=2)
    if args.gpu:
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    losses = []
    val_losses = []
    evals = []
    epoch_loss = 0
    bad = 0

    vis = VisdomClass()
    vis.vis_init("celldetection", args.batch_size)

    iteration = 0
    for epoch in range(args.epochs):
        net.train()
        print("Starting epoch {}/{}.".format(epoch + 1, args.epochs))

        pbar = tqdm(total=number_of_traindata)
        for i, data in enumerate(train_dataset_loader):
            imgs = data["image"]
            true_masks = data["gt"]

            if args.gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(args.batch_size)

            vis.vis_show_result(iteration, loss, masks_pred, imgs, true_masks, args.batch_size)
            iteration += 1
        pbar.close()
        loss = epoch_loss / (number_of_traindata + 1)
        print("Epoch finished ! Loss: {}".format(loss))

        losses.append(loss)
        if epoch % 10 == 0:
            torch.save(net.state_dict(),
                       str(save_weight_path.parent.joinpath("epoch_weight/{:05d}.pth".format(epoch))))
        val_loss = eval_net(net, val_loader, gpu=args.gpu)
        if loss < 0.1:
            print("val_loss: {}".format(val_loss))
            try:
                if min(val_losses) > val_loss:
                    print("update best")
                    torch.save(net.state_dict(), str(args.save_weight_path))
                    bad = 0
                else:
                    bad += 1
                    print("bad ++")
            except ValueError:
                torch.save(net.state_dict(), str(args.save_weight_path))
            val_losses.append(val_loss)
        else:
            print("loss is too large. Continue train")
            val_losses.append(val_loss)
        print("bad = {}".format(bad))
        epoch_loss = 0

        if bad >= 100:
            print("stop running")
            break
    show_graph(losses, val_losses)


if __name__ == "__main__":
    args = parse_args()

    img_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/PHASE").glob("*.tif"))
    gfp_heatmap_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/GFP_heatmap").glob("*.tif"))
    rfp_heatmap_paths = sorted(Path("/home/kazuya/dataset/Riken_2type_cell/RFP_heatmap").glob("*.tif"))

    idx_list = list(range(len(img_paths)))
    import random
    random.shuffle(idx_list)

    args.train_img_path = [img_paths[i] for i in idx_list[:27]]
    args.train_gfp_path = [gfp_heatmap_paths[i] for i in idx_list[:27]]
    args.train_rfp_path = [rfp_heatmap_paths[i] for i in idx_list[:27]]

    args.val_img_path = [img_paths[i] for i in idx_list[27:]]
    args.val_gfp_path = [gfp_heatmap_paths[i] for i in idx_list[27:]]
    args.val_rfp_path = [rfp_heatmap_paths[i] for i in idx_list[27:]]

    # save weight path
    args.save_weight_path = Path("weight/detection_2type/best.pth")

    train = train(args)
