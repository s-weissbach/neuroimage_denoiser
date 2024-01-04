import argparse
from utils.trainfiles import TrainFiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Search Tool")
    # Required argument
    parser.add_argument("--yaml", required=True, help="Path to yaml file")

    parser.add_argument(
        "--path", "-p", required=True, help="Path to folder containing images"
    )
    parser.add_argument(
        "--fileendings",
        "-f",
        required=True,
        nargs="+",
        help="List of file endings to consider",
    )
    # Optional arguments
    parser.add_argument(
        "--kernel_size",
        "-k",
        type=int,
        default=32,
        help="Kernel size for one patch of the image sequence (default: 32)",
    )
    parser.add_argument(
        "--traindir", "-t", required=True, help="Path to outputfolder where the h5 files for training will be stored"
    )
    parser.add_argument(
        "--min_z_score",
        "-z",
        type=int,
        default=4,
        help="Minimum Z score to be considered active patch (default: 4)",
    )
    parser.add_argument(
        "--window_size",
        "-w",
        type=int,
        default=50,
        help="Number of frames used for rolling window z-normalization (default: 50)",
    )
    parser.add_argument(
        "--threads", "-n", type=int, default=1, help="Number of threads (default: 1)"
    )
    parser.add_argument(
        "--fgsplit",
        "-s",
        type=float,
        default=0.5,
        help="Foreground to background split (default: 0.5)",
    )
    args = parser.parse_args()
    yaml_path = args.yaml
    folder_path = args.path
    file_endings = args.fileendings
    kernel_size = args.kernel_size
    min_z_score = args.min_z_score
    window_size = args.window_size
    num_threads = args.threads
    fg_split = args.fgsplit
    train_dir = args.traindir

    trainfiles = TrainFiles(yaml_path, True)

    trainfiles.files_to_traindata(
        directory=folder_path,
        fileendings=file_endings,
        min_z_score=min_z_score,
        kernel_size=kernel_size,
        window_size=window_size,
        n_threads=num_threads,
        foreground_background_split=fg_split,
        train_dir=train_dir
    )


if __name__ == "__main__":
    main()
