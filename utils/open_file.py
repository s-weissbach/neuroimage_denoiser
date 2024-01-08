import numpy as np
import tifffile
import nd2


def open_file(filepath: str) -> np.ndarray:
    """
    Open and read an image file from the specified filepath.

    Parameters:
    - filepath (str): Path to the image file.

    Returns:
    - np.ndarray: NumPy array representing the image data.

    Raises:
    - NotImplementedError: If the file format is not supported.

    Supported file formats:
    - .nd2 (using nd2reader library)
    - .tif, .tiff, .stk (using tifffile library)
    """
    tiff_fileendings = [".tif", ".tiff", ".stk"]
    if filepath.endswith("nd2"):
        return nd2.imread(filepath).astype(np.float64)
    elif any([filepath.endswith(fileending) for fileending in tiff_fileendings]):
        return tifffile.imread(filepath).astype(np.float64)
    else:
        raise NotImplementedError(
            f'Fileformat .{filepath.split(".")[-1]} is currently not implemented. Please change utils/open_file.py'
        )
