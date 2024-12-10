from neuroimage_denoiser.model.unet import UNet
from neuroimage_denoiser.utils.plot import plot_train_loss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from alive_progress import alive_bar
from datetime import datetime


def train(
    model: UNet,
    dataloader: torch.utils.data.DataLoader,
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
    - dataloader (DataLoader): PyTorch DataLoader providing training data.
    - num_epochs (int): Number of training epochs (default is 1).
    - learningrate (float): Learning rate for the optimizer (default is 0.0001).
    - modelpath (str): Filepath to save the trained model (default is "unet.pt").
    - gpu_num (str): GPU device number to use if available (default is "0").
    - history_savepath (str): Filepath to save the training loss history (default is "train_loss.npy").
    - pbar (bool): Whether to show progress bar during training (default is True).
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
            f"The selected loss function ('{lossfunction}') is not available. Select from {list(lossfunctions.keys())}."
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
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        if pbar:
            with alive_bar(len(dataloader)) as bar:
                for i, (data, targets) in enumerate(dataloader):
                    data = data.to(device)
                    targets = targets.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    history.append(loss.item())
                    if i % 10 == 0:
                        print(
                            f"Batch {i+1} (samples {(i+1)*dataloader.batch_size}), Loss: {loss.item():.6f}"
                        )
                    bar()
        else:
            for i, (data, targets) in enumerate(dataloader):
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.append(loss.item())
                if i % 10 == 0:
                    print(
                        f"Batch {i+1} (samples {(i+1)*dataloader.batch_size}), Loss: {loss.item():.6f}"
                    )
    # Save training results
    history = np.array(history)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    plot_train_loss(history, f"{timestamp}_train_loss.pdf")
    np.save(history_savepath, history)
    torch.save(model.state_dict(), modelpath)
