from deep_iglu_denoiser.model.unet import UNet
from deep_iglu_denoiser.utils.dataloader import DataLoader
from deep_iglu_denoiser.utils.normalization import reverse_z_norm
from deep_iglu_denoiser.utils.plot import plot_img, plot_train_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import Union


def train(
    model: UNet,
    dataloader: DataLoader,
    num_epochs: int = 1,
    learningrate: float = 0.0001,
    modelpath: str = "unet.pt",
    history_savepath: str = "train_loss.npy",
    example_img: Union[np.ndarray, None] = None,
    example_img_target: Union[np.ndarray, None] = None,
    example_mean: Union[np.ndarray, None] = None,
    example_std: Union[np.ndarray, None] = None,
    predict_example_every_n_batches: int = 100,
) -> None:
    """
    Train the U-Net model using the specified data loader.

    Parameters:
    - model (UNet): U-Net model to be trained.
    - dataloader (DataLoader): Data loader providing training data.
    - num_epochs (int): Number of training epochs (default is 1).
    - learningrate (float): Learning rate for the optimizer (default is 0.0001).
    - modelpath (str): Filepath to save the trained model (default is "unet.pt").
    - history_savepath (str): Filepath to save the training loss history (default is "train_loss.npy").
    - example_img_path (str): Path to an example image for periodic model predictions (default is "").
    - predict_example_every_n_batches (int): Interval for making model predictions using the example image (default is 100).
    """
    vmin = -np.inf
    vmax = np.inf
    if not example_img is None:
        os.makedirs("example", exist_ok=True)
        example_img = torch.tensor(example_img, dtype=torch.float)  # type: ignore
        if not example_img_target is None:
            vmin = np.min(example_img_target)
            vmax = np.max(example_img_target)
            plot_img(
                example_img_target, f"example/groundtruth.png", vmin=vmin, vmax=vmax
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    criterion = nn.L1Loss()
    history = []
    # Training loop
    for _ in range(num_epochs):
        i = 0
        while not dataloader.epoch_done:
            batch_generated = dataloader.get_batch()
            if not batch_generated:
                break
            data = dataloader.X.to(device)
            targets = dataloader.y.to(device)
            model.train()
            outputs = model(data)
            loss = criterion(outputs, targets)
            history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    f"Batch {i+1} (samples {(i+1)*dataloader.batch_size}), Loss: {loss.item()}"
                )
            if (
                i % predict_example_every_n_batches == 0
                and not example_img is None
                and not example_mean is None
                and not example_std is None
            ):
                model.eval()
                # prevent GPU OOM
                model.to("cpu")
                prediction = model(example_img)
                prediction_np = np.array(prediction.detach())
                prediction_np = prediction_np.reshape(
                    prediction_np.shape[-2], prediction_np.shape[-1]
                )
                prediction_np = reverse_z_norm(prediction_np, example_mean, example_std)
                plot_img(
                    prediction_np, f"example/model_prediction_{i}-batch.png", vmin, vmax
                )
                model.to(device)
            i += 1
        dataloader.shuffle_array()
    history = np.array(history)
    plot_train_loss(history, f"example/train_loss.pdf")
    np.save(history_savepath, history)
    torch.save(model.state_dict(), modelpath)
