import os
from alive_progress import alive_bar
from neuroimage_denoiser.model.modelwrapper import ModelWrapper
from neuroimage_denoiser.utils.copy_folder_structure import copy_folder_structure


def inference(
    path: str,
    modelpath: str,
    directory_mode: False,
    outputpath: str,
    batch_size: int,
    cpu: bool,
    gpu_num: str,
    pbar: bool = True,
    num_frames: int = 5,
) -> None:
    """
    Main function for denoising images using a trained model.

    Args:
        path (str): Path to the input image or directory containing images.
        modelpath (str): Path to the model weights.
        directory_mode (bool): Flag to enable directory mode (True/False).
        outputpath (str): Path to the output directory.
        batch_size (int): Number of frames predicted at once.
    """
    valid_fileendings = [".tif", ".tiff", ".stk", ".nd2"]
    # ensure absolute path
    outputpath = os.path.abspath(outputpath)
    os.makedirs(outputpath, exist_ok=True)
    path = os.path.abspath(path)
    # initalize model
    model = ModelWrapper(modelpath, batch_size, cpu, gpu_num, num_frames)
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
    if pbar:
        with alive_bar(len(filelist)) as bar:
            for filepath, outpath in zip(filelist, outputpaths):
                filename = os.path.splitext(os.path.basename(filepath))[0]
                outfilepath = os.path.join(outpath, f"{filename}_denoised.tif")
                if os.path.exists(outfilepath):
                    print(
                        f"Skipped {filename}, because file already exists ({outfilepath})."
                    )
                    bar()
                    continue
                try:
                    model.denoise_img(filepath)
                    model.write_denoised_img(outfilepath)
                    print(
                        f"Saved image ({os.path.basename(filepath)}) as: {outfilepath}"
                    )
                except Exception as error:
                    print(f"Skipped {filepath}, due to an unexpected error:")
                    print(error)
                bar()
    else:
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
