import argparse
import yaml

from neuroimage_denoiser.utils.trainfiles import TrainFiles
from neuroimage_denoiser.utils.dataloader import DataLoader
from neuroimage_denoiser.model.unet import UNet
from neuroimage_denoiser.model.train import train
from neuroimage_denoiser.model.denoise import inference


def main():
    parser = argparse.ArgumentParser(description="iGlu Denoiser")
    subparsers = parser.add_subparsers(dest="mode")

    pre_training_p = subparsers.add_parser("prepare_training")
    pre_training_p.add_argument(
        "--path", "-p", required=True, help="Path to folder containing images"
    )
    pre_training_p.add_argument(
        "--fileendings",
        "-f",
        required=True,
        nargs="+",
        help="List of file endings to consider",
    )
    # Optional arguments
    pre_training_p.add_argument(
        "--crop_size",
        "-c",
        type=int,
        default=32,
        help="Crop size used during training (default: 32)",
    )
    pre_training_p.add_argument(
        "--roi_size",
        type=int,
        default=4,
        help="Expected ROI size; assumes for detection square of (roi_size x roi_size) (default: 8)",
    )
    pre_training_p.add_argument(
        "--h5",
        required=True,
        help="Path to outputpath of the h5 file that will be created",
    )
    pre_training_p.add_argument(
        "--min_z_score",
        "-z",
        type=float,
        default=2.0,
        help="Minimum Z score to be considered active patch (default: 2)",
    )
    pre_training_p.add_argument(
        "--window_size",
        "-w",
        type=int,
        default=50,
        help="Number of frames used for rolling window z-normalization (default: 50)",
    )
    pre_training_p.add_argument(
        "--n_frames", "-n",
        required=True,
        type=int,
        help="Number of frames around the target frame used for denoising."
    )
    pre_training_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing h5 file. If false, data will be appended. (default: False)",
    )
    pre_training_p.add_argument(
        "--memory_optimized",
        action="store_true",
        help="Utilize optimized memory mode. Trades speed for lower memory usage",
    )
    # Training
    train_p = subparsers.add_parser("train")
    train_p.add_argument(
        "--trainconfigpath", "-p", required=True, help="Path to train config YAML file"
    )
    # Denoise / Inference
    denoise_p = subparsers.add_parser("denoise")
    denoise_p.add_argument(
        "--path", "-p", type=str, required=True, help="Specify the path."
    )
    denoise_p.add_argument(
        "--modelpath", "-m", type=str, required=True, help="Path to modelweights."
    )
    denoise_p.add_argument(
        "--n_frames", "-n", type=int, required=True, help="Number of frames around the target frame used for denoising."
    )
    denoise_p.add_argument(
        "--directory_mode", "-d", action="store_true", help="Enable directory mode."
    )
    denoise_p.add_argument(
        "--outputpath", "-o", type=str, required=True, help="Specify the output path."
    )
    denoise_p.add_argument(
        "--cpu", action="store_true", help="Force CPU and not use GPU."
    )

    args = parser.parse_args()
    if args.mode == "prepare_training":
        trainfiles = TrainFiles(
            fileendings=args.fileendings,
            min_z_score=args.min_z_score,
            crop_size=args.crop_size,
            roi_size=args.roi_size,
            output_h5_file=args.h5,
            window_size=args.window_size,
            foreground_background_split=args.fgsplit,
            overwrite=args.overwrite,
            n_frames=args.n_frames
        )
        # gather train data
        trainfiles.files_to_traindata(
            directory=args.path,
            memory_optimized=args.memory_optimized,
        )
    # training
    elif args.mode == "train":
        trainconfigpath = args.trainconfigpath
        # parse train config file
        with open(trainconfigpath, "r") as f:
            trainconfig = yaml.safe_load(f)
        modelpath = trainconfig["modelpath"]
        h5 = trainconfig["train_h5"]
        batch_size = trainconfig["batch_size"]
        learning_rate = trainconfig["learning_rate"]
        num_epochs = trainconfig["num_epochs"]
        n_frames = trainconfig["n_frames"]
        dataloader = DataLoader(
            h5,
            batch_size,
            n_frames
        )
        model = UNet(n_frames)
        train(model, dataloader, num_epochs, learning_rate, modelpath)
    # denoising / inference
    elif args.mode == "denoise":
        inference(
            args.path,
            args.modelpath,
            args.directory_mode,
            args.outputpath,
            args.cpu,
            args.n_frames
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()