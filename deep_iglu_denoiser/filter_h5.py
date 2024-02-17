from alive_progress import alive_bar

import h5py
from scipy.ndimage import uniform_filter
import numpy as np
import argparse


def main(input_h5: str, output_h5: str, min_z: float, roi_size: int) -> None:
    # input
    f_in = h5py.File(input_h5, "r")
    # "/mnt/nvme2/iGlu_train_data/iglu_train_data_cropsize32_roisize4_stim_z2_filtered.h5"
    f_out = h5py.File(output_h5, "w")
    idx = 0
    num_samples = f_in.__len__()
    with alive_bar(num_samples) as bar:
        for i in range(num_samples):
            frame = np.array(f_in.get(str(i)))
            mean_frame = uniform_filter(frame, roi_size, mode="constant")
            if np.any(mean_frame > min_z):
                f_out.create_dataset(str(idx), data=frame)
                idx += 1
                if idx % 1_000 == 0:
                    print(f"Wrote {idx} files to the filtered h5-file.")
            bar()
    f_out.close()
    f_in.close()
    print(f"Kept {idx} of {num_samples} examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for VineSeg")
    parser.add_argument(
        "--input_h5", "-i", type=str, help="Path to the input H5 file", required=True
    )
    parser.add_argument(
        "--output_h5", "-o", type=str, help="Path to the output H5 file", required=True
    )
    parser.add_argument(
        "--min_z", "-z", type=float, help="Minimum Z value", required=True
    )
    parser.add_argument(
        "--roi_size",
        "-r",
        type=int,
        help="Size of the Region of Interest (ROI)",
        required=True,
    )
    args = parser.parse_args()

    main(args.input_h5, args.output_h5, args.min_z, args.roi_size)
