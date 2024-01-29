import argparse
import os
from alive_progress import alive_bar
from deep_iglu_denoiser.model.modelwrapper import ModelWrapper
from deep_iglu_denoiser.utils.copy_folder_structure import copy_folder_structure


def main(
    path: str, modelpath: str, directory_mode: str, outputpath: str, batch_size: int, cpu: bool
) -> None:
    """
    Main function for denoising images using a trained model.

    Args:
        path (str): Path to the input image or directory containing images.
        modelpath (str): Path to the model weights.
        directory_mode (str): Flag to enable directory mode (True/False).
        outputpath (str): Path to the output directory.
        batch_size (int): Number of frames predicted at once.
    """
    valid_fileendings = [".tif", ".tiff", ".stk", ".nd2"]
    # ensure absolute path
    outputpath = os.path.abspath(outputpath)
    path = os.path.abspath(path)
    # initalize model
    model = ModelWrapper(modelpath, batch_size, cpu)
    if directory_mode:
        # preserver original folderstructure
        copy_folder_structure(path, outputpath)
        filelist = []
        outputpaths = []
        for folderpath, _, filenames in os.walk(path):
            for filename in filenames:
                if not any([filename.endswith(ending) for ending in valid_fileendings]):
                    continue
                filelist.append(os.path.join(folderpath, filename))
                # preserver original folderstructure
                if folderpath == path:
                    outputpaths.append(outputpath)
                else:
                    outputpaths.append(
                        os.path.join(
                            outputpath, folderpath.split(f"{path}" + os.path.sep)[1]
                        )
                    )
    else:
        if not any([path.endswith(ending) for ending in valid_fileendings]):
            raise NotImplementedError(
                f"Fileending .{path.split('.')[-1]} is currently not implemented."
            )
        filelist = [path]
        outputpaths = [outputpath]

    with alive_bar(len(filelist)) as bar:
        for filepath, outpath in zip(filelist, outputpaths):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            outfilepath = os.path.join(outpath, f"{filename}_denoised.tif")
            if os.path.exists(outfilepath):
                print(
                    f"Skipped {filename}, because file already exists ({outfilepath})."
                )
                continue
            model.denoise_img(filepath)
            model.write_denoised_img(outfilepath)
            print(f"Saved image ({os.path.basename(filepath)}) as: {outfilepath}")
            bar()


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Custom Argument Parser")

    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Specify the path."
    )
    parser.add_argument(
        "--modelpath", "-m", type=str, required=True, help="Path to modelweights."
    )
    parser.add_argument(
        "--directory_mode", "-d", action="store_true", help="Enable directory mode."
    )
    parser.add_argument(
        "--outputpath", "-o", type=str, required=True, help="Specify the output path."
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=1,
        help="Number of frames that are predicted at once.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU and not use GPU."
    )

    args = parser.parse_args()

    # Check if the specified path exists
    if not os.path.exists(args.path):
        raise argparse.ArgumentError(None, f"Path '{args.path}' does not exist.")

    # Check if the specified path is a directory if directory_mode is enabled
    if args.directory_mode and not os.path.isdir(args.path):
        raise argparse.ArgumentError(None, f"Path '{args.path}' is not a directory.")

    # Check if the specified path is a file if directory_mode is not enabled
    if not args.directory_mode and not os.path.isfile(args.path):
        raise argparse.ArgumentError(None, f"Path '{args.path}' is not a file.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.outputpath, exist_ok=True)

    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    # Call the main function with the parsed arguments
    main(
        args.path, args.modelpath, args.directory_mode, args.outputpath, args.batchsize, args.cpu
    )
