import argparse
from deep_iglu_denoiser.utils.open_file import open_file
from deep_iglu_denoiser.utils.normalization import z_norm
from deep_iglu_denoiser.utils.dataloader import DataLoader
from deep_iglu_denoiser.model.unet import UNet
from deep_iglu_denoiser.model.train import train

import torch
import yaml
import numpy as np


def main() -> None:
    """
    Main function to start training
    """
    parser = argparse.ArgumentParser(description="Start training of UNet for denoising")

    parser.add_argument(
        "--trainconfigpath", "-p", required=True, help="Path to train config YAML file"
    )
    # parse args
    args = parser.parse_args()
    trainconfigpath = args.trainconfigpath

    # parse train config file
    with open(trainconfigpath, "r") as f:
        trainconfig = yaml.safe_load(f)
    modelpath = trainconfig["modelpath"]
    train_h5 = trainconfig["train_h5"]
    batch_size = trainconfig["batch_size"]
    learning_rate = trainconfig["learning_rate"]
    num_epochs = trainconfig["num_epochs"]
    pre_frames = trainconfig["pre_frames"]
    post_frames = trainconfig["post_frames"]

    dataloader = DataLoader(train_h5, batch_size, pre_frames, post_frames)
    model = UNet(pre_frames+post_frames)

    train(model, dataloader, num_epochs, learning_rate, modelpath)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU ready")
    main()
