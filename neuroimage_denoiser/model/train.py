from neuroimage_denoiser.model.unet import UNet
from torch.utils.data import DataLoader
from neuroimage_denoiser.utils.plot import plot_train_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from alive_progress import alive_bar
from datetime import datetime


def train(
    model: UNet,
    dataloader: DataLoader,
    num_epochs: int = 1,
    learningrate: float = 0.0001,
    lossfunction: str = "L1",
    modelpath: str = "unet.pt",
    gpu_num: str = "0",
    history_savepath: str = "train_loss.npy",
    pbar: bool = True,
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
    lossfunctions = {
        "L1": nn.L1Loss(),
        "Smooth-L1": nn.SmoothL1Loss(),
        "MSE": nn.MSELoss(),
        "Crossentropy": nn.CrossEntropyLoss(),
        "Huber": nn.HuberLoss(),
    }
    if lossfunction not in lossfunctions.keys():
        raise NotImplementedError(
            f"The selected loss function ('{lossfunction}) is not available. Select from {list(lossfunctions.keys())}."
        )
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING! Training on the CPU can be very (!) time consuming.")
    else:
        print("GPU ready")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    criterion = lossfunctions[lossfunction]
    history = []
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        if pbar:
            with alive_bar(len(dataloader)) as bar:
                for i, (data, targets) in enumerate(dataloader):
                    data = data.to(device)
                    targets = targets.to(device)
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
                    bar()
        else:
            for i, (data, targets) in enumerate(dataloader):
                data = data.to(device)
                targets = targets.to(device)

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
    history = np.array(history)
    plot_train_loss(
        history, datetime.now().strftime("%Y-%m-%d_%H:%M:%S_train_loss.pdf")
    )
    np.save(history_savepath, history)
    torch.save(model.state_dict(), modelpath)
