import matplotlib.pyplot as plt
import numpy as np


def plot_img(
    img: np.ndarray, savepath: str = "", vmin: float = -np.inf, vmax: float = np.inf
) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    if vmin != -np.inf:
        vmin = vmin
    else:
        vmin = np.min(img)
    if vmax != np.inf:
        vmax = vmax
    else:
        vmax = np.max(img)
    ax.imshow(img, cmap="Greys", vmin=vmin, vmax=vmax)
    for orientation in ["top", "bottom", "left", "right"]:
        ax.spines[orientation].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if savepath != "":
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def plot_train_loss(loss: np.ndarray[np.float64], savepath: str) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    ax.plot(loss,color='#293241',lw=2)
    for orientation in ["top", "right"]:
        ax.spines[orientation].set_visible(False)
    for orientation in ["bottom", "left"]:
        ax.spines[orientation].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=15,width=3,length=10)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches = 'tight')
    plt.clf()
    plt.cla()
    plt.close()
