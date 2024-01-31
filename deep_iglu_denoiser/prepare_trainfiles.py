import argparse
from deep_iglu_denoiser.utils.trainfiles import TrainFiles


def main() -> None:
    """
    Main function for preparing train files used for training a model.
    """
    parser = argparse.ArgumentParser(description="Image Search Tool")
    # Required argument
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
        "--crop_size",
        "-c",
        type=int,
        default=32,
        help="Crop size used during training (default: 32)",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=4,
        help="Expected ROI size; assumes for detection square of (roi_size x roi_size) (default: 8)",
    )
    parser.add_argument(
        "--trainh5",
        "-t",
        required=True,
        help="Path to outputpath of the h5 file that will be created",
    )
    parser.add_argument(
        "--min_z_score",
        "-z",
        type=float,
        default=2.0,
        help="Minimum Z score to be considered active patch (default: 2)",
    )
    parser.add_argument(
        "--before",
        type=int,
        default=0,
        help="Number of frames to add before a detected event, to also train to reconstruct the typical raise of the sensor. (default: 0)",
    )
    parser.add_argument(
        "--after",
        type=int,
        default=0,
        help="Number of frames to add after a detected event, to also train to reconstruct the typical decay of the sensor. (default: 0)",
    )
    parser.add_argument(
        "--activitymap",
        action="store_true",
        help="Search for active ROI without prior information.",
    )
    parser.add_argument(
        "--stimulationframes",
        nargs="+",
        help="List of frames that were stimulated and thus activity is expected",
    )
    parser.add_argument(
        "--n_frames", type=int, help="Number of frames to include after stimulation."
    )
    parser.add_argument(
        "--window_size",
        "-w",
        type=int,
        default=50,
        help="Number of frames used for rolling window z-normalization (default: 50)",
    )
    parser.add_argument(
        "--fgsplit",
        "-s",
        type=float,
        default=0.5,
        help="Foreground to background split (default: 0.5)",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite existing h5 file. If false, data will be appended. (default: False)",
    )
    parser.add_argument(
        "--memory_optimized",
        type=bool,
        default=False,
        help="Utilize optimized memory mode. Trades speed for lower memory usage",
    )
    # parse arguments
    args = parser.parse_args()
    folder_path = args.path
    file_endings = args.fileendings
    crop_size = args.crop_size
    roi_size = args.roi_size
    min_z_score = args.min_z_score
    frames_before_event = args.before
    frames_after_event = args.after
    window_size = args.window_size
    activitymap = args.activitymap
    stimulationframes = (
        [int(frame) for frame in args.stimulationframes]
        if args.stimulationframes
        else []
    )
    n_frames = args.n_frames
    fg_split = args.fgsplit
    output_h5_file = args.trainh5
    overwrite = args.overwrite
    memory_optimized = args.memory_optimized

    # initalize TrainFiles class
    trainfiles = TrainFiles(
        fileendings=file_endings,
        min_z_score=min_z_score,
        frames_before_event=frames_before_event,
        frames_after_event=frames_after_event,
        crop_size=crop_size,
        roi_size=roi_size,
        output_h5_file=output_h5_file,
        window_size=window_size,
        activitymap=activitymap,
        stimulationframes=stimulationframes,
        n_frames=n_frames,
        foreground_background_split=fg_split,
        overwrite=overwrite,
    )

    # gather train data
    trainfiles.files_to_traindata(
        directory=folder_path,
        memory_optimized=memory_optimized,
    )


if __name__ == "__main__":
    main()
