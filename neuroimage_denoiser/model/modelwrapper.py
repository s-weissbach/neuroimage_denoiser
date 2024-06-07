from neuroimage_denoiser.model.unet import UNet
import neuroimage_denoiser.utils.normalization as normalization
from neuroimage_denoiser.utils.write_file import write_file
from neuroimage_denoiser.utils.open_file import open_file
from neuroimage_denoiser.utils.convert import float_to_uint
import torch
import numpy as np


class ModelWrapper:
    """
    A wrapper class for a U-Net model used for denoising 3D image sequences.

    This class provides methods to load pre-trained weights, normalize input images,
    perform denoising on image sequences, and write the denoised output to a file.

    Attributes:
        weights (str): Path to the pre-trained weights.
        batch_size (int): Number of frames to process in each batch.
        cpu (bool): Flag to force CPU usage, even if a GPU is available.
        device (torch.device): Device to use for computations (GPU or CPU).
        model (UNet): The U-Net model instance.
        denoised_img (np.ndarray): The denoised image sequence.
        img (np.ndarray): The input image sequence.
        img_height (int): Height of the input image sequence.
        img_width (int): Width of the input image sequence.
        img_mean (np.ndarray): Mean of the input image sequence along the z-axis.
        img_std (np.ndarray): Standard deviation of the input image sequence along the z-axis.
    """

    def __init__(self, weights: str, batch_size: int, cpu: bool) -> None:
        """
        Initialize the ModelWrapper instance.

        Args:
            weights (str): Path to the pre-trained weights.
            batch_size (int): Number of frames to process in each batch.
            cpu (bool): Flag to force CPU usage, even if a GPU is available.
        """
        # initalize model
        self.batch_size = batch_size
        # check for GPU, use CPU otherwise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if flag cpu is set, use cpu regardless of available GPU
        if cpu:
            self.device = "cpu"
        self.model = UNet(1)
        self.load_weights(weights)
        self.model.to(self.device)
        # initalize image
        self.denoised_img = np.empty((0, 0, 0))
        self.img = np.empty((0, 0, 0))
        self.img_height = -1
        self.img_width = -1
        self.img_mean = np.empty((0, 0))
        self.img_std = np.empty((0, 0))

    def load_weights(self, weights: str) -> None:
        """
        Load pre-trained weights into the U-Net model.

        Args:
            weights (str): Path to the pre-trained weights file.
        """
        if self.device == "cpu":
            self.model.load_state_dict(
                torch.load(weights, map_location=torch.device("cpu"))
            )
        else:
            self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def normalize_img(self) -> None:
        """
        Normalize the input image sequence using z-score normalization.

        This method computes the mean and standard deviation along the z-axis
        of the input image sequence and performs z-score normalization using
        these values.
        """
        self.img_mean: np.ndarray = np.mean(self.img, axis=0)
        self.img_std: np.ndarray = np.std(self.img, axis=0)
        # normalization
        self.img: np.ndarray = normalization.z_norm(
            self.img, self.img_mean, self.img_std
        )

    def get_prediction_frames(self, from_frame: int) -> torch.Tensor:
        """
        Extract frames around the target frame for making predictions.

        This method extracts a batch of frames from the input image sequence,
        starting from the specified frame index. The extracted frames are
        reshaped and converted to a PyTorch tensor for input to the U-Net model.

        Args:
            from_frame (int): Index of the starting frame for the batch.

        Returns:
            torch.Tensor: Input tensor for the U-Net model, containing a batch of frames.
        """
        to_frame = min(len(self.img), from_frame + self.batch_size)
        # extract frames
        X = self.img[from_frame:to_frame]
        # reshape to batch size 1
        X = X.reshape(
            min(self.batch_size, to_frame - from_frame),
            1,
            self.img_height,
            self.img_width,
        )
        return torch.tensor(X, dtype=torch.float)

    def inference(self) -> list[np.ndarray]:
        """
        Perform inference on the input image sequence using the U-Net model.

        This method processes the input image sequence in batches using the
        U-Net model and returns the denoised frames as a list of NumPy arrays.

        Returns:
            list[np.ndarray]: List of denoised frames.
        """
        denoised_image_sequence = []
        for from_frame in range(0, self.img.shape[0], self.batch_size):
            X = self.get_prediction_frames(from_frame).to(self.device)
            y_pred = np.array(self.model(X).detach().to("cpu"))
            for denoised_frame in y_pred:
                denoised_image_sequence.append(
                    denoised_frame.reshape(self.img_height, self.img_width)
                )
        return denoised_image_sequence

    def denoise_img(self, img_path: str) -> None:
        """
        Denoise an image sequence using the U-Net model.

        This method loads and normalizes the input image sequence, performs
        inference using the U-Net model, and stores the denoised image sequence
        in the `denoised_img` attribute.

        Args:
            img_path (str): Path to the image sequence file.
        """

        self.img: np.ndarray = open_file(img_path)
        _, self.img_height, self.img_width = self.img.shape
        self.normalize_img()
        denoised_image_sequence = self.inference()
        self.denoised_img = normalization.reverse_z_norm(
            np.array(denoised_image_sequence), self.img_mean, self.img_std
        )
        # tiff format is based on uint16 -> cast
        self.denoised_img = float_to_uint(self.denoised_img)

    def write_denoised_img(self, outpath: str) -> None:
        """
        Write the denoised image sequence to a file.

        This method writes the denoised image sequence stored in the `denoised_img`
        attribute to the specified output file path.

        Args:
            outpath (str): Path to the output file where the denoised image sequence
                           should be written.

        Raises:
            AssertionError: If the `denoised_img` attribute is empty, indicating that
                            the image sequence has not been denoised yet.
        """
        if self.denoised_img.shape[0] == 0:
            raise AssertionError(
                f"Before writing a denoised image, first denoise image. Use <ModelWrapper>.denoise_img(<path/to/input_image>)."
            )
        write_file(self.denoised_img, outpath)
