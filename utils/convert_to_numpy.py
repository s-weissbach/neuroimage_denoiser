import tifffile
from utils.trainfiles import find_files
import argparse
import os
import numpy as np
from tqdm import tqdm


def to_train_data(
    img_sequence: np.ndarray, sequence_len: int, outpath: str, filename: str
):
    """
    Process an image sequence to create training data with a rolling window.

    Parameters:
    - img_sequence (np.ndarray): The input image sequence with shape (frames, height, width).
    - sequence_len (int): The number of frames before and after the target frame.

    Returns:
    - tuple[list, int]: A tuple containing the updated data dictionary and the new index.
    """
    X_train = []
    y_train = []
    for target_frame in range(sequence_len, img_sequence.shape[0] - sequence_len - 1):
        y = img_sequence[target_frame]
        X = np.append(
            img_sequence[target_frame - sequence_len : target_frame],
            img_sequence[target_frame + 1 : target_frame + sequence_len + 1],
            axis=0,
        )
        X_train.append(X)
        y_train.append(y)
    print("save x")
    np.save(os.path.join(outpath, f"{filename}_xtrain.npy"), np.array(X_train))
    print("save y")
    np.save(os.path.join(outpath, f"{filename}_ytrain.npy"), np.array(y_train))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process images and optionally add data to an existing H5 file."
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="Path to the root directory."
    )
    parser.add_argument(
        "--fileendings", nargs="+", required=True, help="List of file endings."
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=2,
        help="Number of frames before and after the target frame used to predict the target frame.",
    )
    parser.add_argument(
        "-o", "--outpath", required=True, help="Output path for the h5 file."
    )
    args = parser.parse_args()

    directory = args.directory
    fileendings = args.fileendings
    sequence_length = args.sequence_length
    outpath = args.outpath

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"The specified directory '{directory}' does not exist."
        )
    filelist = find_files(directory, fileendings)
    for file in tqdm(filelist, total=len(filelist), desc="processing images"):
        img_sequence = tifffile.imread(file)
        to_train_data(
            img_sequence,
            sequence_length,
            outpath,
            os.path.splitext(os.path.basename(file))[0],
        )


if __name__ == "__main__":
    main()
