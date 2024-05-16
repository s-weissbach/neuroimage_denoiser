from neuroimage_denoiser.model.unet import UNet
from neuroimage_denoiser.utils.dataloader import DataLoader
from neuroimage_denoiser.utils.plot import plot_train_loss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from alive_progress import alive_bar


def train(
    model: UNet,
    dataloader: DataLoader,
    num_epochs: int = 1,
    learningrate: float = 0.0001,
    modelpath: str = "unet.pt",
    history_savepath: str = "train_loss.npy",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print('WARNING! No GPU detected. Training on CPU is not recommended and will take very long.')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    criterion = nn.L1Loss()
    history = []
    # Training loop

    for _ in range(num_epochs):
        i = 0
        with alive_bar(dataloader.__len__()) as bar:
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
                bar()
                i += 1
        dataloader.shuffle_array()
    history = np.array(history)

    plot_train_loss(history, f"train_loss.pdf")
    np.save(history_savepath, history)
    torch.save(model.state_dict(), modelpath)
