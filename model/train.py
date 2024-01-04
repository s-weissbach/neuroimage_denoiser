from model.unet import UNet
from utils.dataloader import DataLoader
from utils.open_file import open_file
import utils.normalization as normalization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt


def train(
    model: UNet,
    dataloader: DataLoader,
    num_epochs: int = 1,
    learningrate: float = 0.0001,
    modelpath: str = "unet.pt",
    history_savepath: str = "train_loss.npy",
    example_img_path: str = "",
    predict_example_every_n_batches: int = 100
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
    if example_img_path != "":
        os.makedirs('example',exist_ok=True)
        example_img = open_file(example_img_path)
        mean = example_img.mean(axis=0)
        std = example_img.std(axis=0)
        example_img = normalization.z_norm(
                example_img[0 : dataloader.n_pre + dataloader.n_post+2], mean, std
            )
        example_img = np.delete(example_img, dataloader.n_pre, axis=0).reshape(1,example_img.shape[0],example_img.shape[1],example_img.shape[2])
        example_img = torch.tensor(example_img,dtype=torch.float)
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
            if i % predict_example_every_n_batches == 0 and example_img_path != '':
                example_img.to(device)
                model.eval()
                prediction = np.array(model(example_img).detach().to('cpu'))
                prediction = prediction.reshape(prediction.shape[-2],prediction.shape[-1])
                fig,ax = plt.subplots()
                fig.set_size_inches(5,5)
                ax.imshow(prediction, 'Greys')
                for orientation in ['top','bottom','left','right']:
                    ax.spines[orientation].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(f'example/model_prediction_{i}-batch.png')
                plt.clf()
                example_img.to('cpu')
            i += 1
        dataloader.shuffle_array()
    history = np.array(history)
    np.save(history_savepath, history)
    torch.save(model.state_dict(), modelpath)
