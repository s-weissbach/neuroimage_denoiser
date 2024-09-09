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

    def __init__(self, weights: str, batch_size: int, cpu: bool, gpu_num: str) -> None:
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
        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        # if flag cpu is set, use cpu regardless of available GPU
        if cpu:
            self.device = "cpu"
        self.model = UNet(1)
        self.load_weights(weights)
        self.model.to(self.device)
        # initalize image
        self.img_tensor = None
        self.img_shape = None
        self.img_mean = None
        self.img_std = None

    def load_weights(self, weights: str) -> None:
        """
        Load pre-trained weights into the U-Net model.

        Args:
            weights (str): Path to the pre-trained weights file.
        """
        self.model.load_state_dict(
                torch.load(weights, map_location=torch.device(self.device))
        )
        self.model.eval()

    def normalize_img(self, img: np.ndarray) -> None:
        """
        Normalize the input image sequence using z-score normalization.

        This method computes the mean and standard deviation along the z-axis
        of the input image sequence and performs z-score normalization using
        these values.
        """
        self.img_std = np.std(img,axis=0)
        self.img_mean = np.mean(img,axis=0)
        # normalization
        img: torch.tensor = normalization.z_norm(
            img, self.img_mean, self.img_std
        )
        return img

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
        to_frame = min(len(self.img_tensor), from_frame + self.batch_size)
        # extract frames
        X = self.img_tensor[from_frame:to_frame]
        # reshape to batch size 1
        X = X.reshape(
            min(self.batch_size, to_frame - from_frame),
            1,
            self.img_height,
            self.img_width,
        )
        return X
    
    def inference(self) -> torch.Tensor:
        """
        Perform inference on the input image sequence using the U-Net model.

        This method processes the input image sequence in batches using the
        U-Net model and returns the denoised frames as a list of NumPy arrays.

        Returns:
            list[np.ndarray]: List of denoised frames.
        """
        denoised_image_sequence = torch.zeros(
            (self.img_tensor.shape[0], self.img_height, self.img_width),
             device='cpu',
             dtype=torch.float32
        )
        for from_frame in range(0, self.img_tensor.shape[0], self.batch_size):
            to_frame = min(from_frame + self.batch_size, self.img_tensor.shape[0])
            current_batch_size = to_frame - from_frame
            X = self.get_prediction_frames(from_frame).to(self.device)
            with torch.no_grad():
                y_pred = self.model(X.type(torch.float))
            denoised_image_sequence[from_frame:to_frame] = y_pred.cpu().reshape(current_batch_size, self.img_height, self.img_width)
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
        img: np.ndarray = open_file(img_path)
        _, self.img_height, self.img_width = img.shape
        img: np.ndarray = self.normalize_img(img)
        self.img_tensor: torch.tensor = torch.tensor(img, dtype=torch.float)
        denoised_image_sequence = np.array(self.inference())
        self.denoised_img = normalization.reverse_z_norm(
            denoised_image_sequence, self.img_mean, self.img_std
        )
        # tiff format is based on uint16 -> cast
        self.denoised_img = float_to_uint(denoised_image_sequence)

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
        if not hasattr(self, 'denoised_img') or self.denoised_img.size == 0:
            raise ValueError("No denoised image available. Run denoise_img first.")
        try:
            write_file(self.denoised_img, outpath)
        except Exception as e:
            raise RuntimeError(f"Failed to write denoised image to {outpath}: {str(e)}")
