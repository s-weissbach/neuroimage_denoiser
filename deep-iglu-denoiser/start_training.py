import argparse
from utils.open_file import open_file
from utils.normalization import z_norm
from utils.dataloader import DataLoader
from model.unet import UNet
from model.train import train

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
    train_h5 =trainconfig["train_h5"]
    batch_size = trainconfig["batch_size"]
    learning_rate = trainconfig["learning_rate"]
    num_epochs = trainconfig["num_epochs"]
    path_example_img = trainconfig["path_example_img"]
    target_frame_example_img = trainconfig["target_frame_example_img"]
    predict_every_n_batches = trainconfig["predict_every_n_batches"]
    noise_center = trainconfig["noise_center"]
    noise_scale = trainconfig["noise_scale"]

    dataloader = DataLoader(train_h5, batch_size, noise_center, noise_scale)
    model = UNet(1)

    if path_example_img == "":
        train(model, dataloader, num_epochs, learning_rate, modelpath)
        
    else:
        example_img = open_file(path_example_img)
        example_img_target_frame = example_img[target_frame_example_img]
        mean = np.mean(example_img, axis=0)
        std = np.std(example_img, axis=0)
        example_img = z_norm(example_img, mean, std)
        example_img_pred_frames = example_img[
            target_frame_example_img
        ]
        example_img_pred_frames = example_img_pred_frames.reshape(1, 1, example_img.shape[-2], example_img.shape[-1])
        train(
            model,
            dataloader,
            num_epochs,
            learning_rate,
            modelpath,
            example_img=example_img_pred_frames,
            example_img_target=example_img_target_frame,
            example_mean=mean,
            example_std=std,
            predict_example_every_n_batches=predict_every_n_batches,
        )
        


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU ready")
    else:
        print("Warning: only CPU found")
    main()
