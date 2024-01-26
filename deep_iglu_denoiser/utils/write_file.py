import numpy as np
import tifffile


def write_file(img: np.ndarray, filepath: str) -> None:
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

    if any([filepath.endswith(fileending) for fileending in tiff_fileendings]):
        tifffile.imwrite(filepath, img)
    else:
        raise NotImplementedError(
            f'Fileformat .{filepath.split(".")[-1]} is currently not implemented. Please change utils/open_file.py'
        )
