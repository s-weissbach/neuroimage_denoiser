import numpy as np
import tifffile
import nd2
from neuroimage_denoiser.utils.convert import uint_to_float


def open_file(filepath: str) -> np.ndarray:
    """
    Open and read an image file from the specified filepath.

    Parameters:
    - filepath (str): Path to the image file.

    Returns:
    - np.ndarray[np.float64]: NumPy array representing the image data.

    Raises:
    - NotImplementedError: If the file format is not supported.

    Supported file formats:
    - .nd2 (using nd2reader library)
    - .tif, .tiff, .stk (using tifffile library)
    """
    tiff_fileendings = [".tif", ".tiff", ".stk"]
    if filepath.endswith("nd2"):
        return uint_to_float(nd2.imread(filepath))
    elif any([filepath.endswith(fileending) for fileending in tiff_fileendings]):
        return uint_to_float(tifffile.imread(filepath))
    else:
        raise NotImplementedError(
            f'Fileformat .{filepath.split(".")[-1]} is currently not implemented. Please change utils/open_file.py'
        )
