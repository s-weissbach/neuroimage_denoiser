from model.unet import UNet
import utils.normalization as normalization
import torch
import numpy as np
import tifffile


class ModelWrapper:
    def __init__(self, weights: str, n_pre: int, n_post: int) -> None:
        # initalize model
        self.n_pre = n_pre
        self.n_post = n_post
        # check for GPU, use CPU otherwise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(self.n_pre + self.n_post)
        self.load_weights(weights)
        self.model.to(self.device)
        # initalize image
        self.img = np.empty
        self.img_height = -1
        self.img_width = -1
        self.img_mean = np.nan
        self.img_std = np.nan

    def load_weights(self, weights: str) -> None:
        self.model.load_state_dict(torch.load(weights))

    def load_img(self, img_path: str) -> None:
        self.img = tifffile.imread(img_path)
        _, self.img_height, self.img_width = self.img.shape
        # normalization
        self.img_mean = np.mean(self.img)
        self.img_std = np.std(self.img)
        self.img: np.ndarray = normalization.z_norm(
            self.img, self.img_mean, self.img_std
        )

    def get_prediction_frames(self, target: int) -> torch.Tensor:
        # extract frames
        X = self.img[target - self.n_pre - 1 : target + self.n_post]  # ignore: warning
        # remove target frame
        X = np.delete(X, self.n_pre)
        # reshape to batch size 1
        X = X.reshape(1, self.n_pre + self.n_post, self.img_height, self.img_width)
        return torch.tensor(X, dtype=torch.float)

    def denoise_img(self, img_path: str) -> np.ndarray:
        denoised_image_sequence = []
        self.load_img(img_path)
        for target in range(self.n_pre + 1, len(self.img) - self.n_post):
            X = self.get_prediction_frames(target).to(self.device)
            y_pred = np.array(self.model(X).detach().to("cpu")).reshape(
                self.img_height, self.img_width
            )
            denoised_image_sequence.append(y_pred)
        y_pred_grey_vals = normalization.reverse_z_norm(
            np.array(denoised_image_sequence), self.img_mean, self.img_std
        )
        return y_pred_grey_vals
