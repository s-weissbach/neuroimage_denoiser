import numpy as np
import tifffile
import nd2

def open_file(filepath: str) -> np.ndarry:
    tiff_fileendings = ['.tif', '.tiff', '.stk']
    if filepath.endswith('nd2'):
        return nd2.imread(filepath)
    elif np.any([filepath.endswith(fileending) for fileending in tiff_fileendings]):
        return tifffile.imread(filepath)
    else:
        raise NotImplementedError(f'Fileformat .{filepath.split(".")[-1]} is currently not implemented. Please change utils/open_file.py')