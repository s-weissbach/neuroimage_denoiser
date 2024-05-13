from neuroimage_denoiser.model.train import train
from neuroimage_denoiser.model.unet import UNet
from neuroimage_denoiser.utils.dataloader import DataLoader
from neuroimage_denoiser.utils.evaluate_model import evaluate, raw_evaluate

import os
from alive_progress import alive_bar
import yaml
import json


def gridsearch_train(trainconfigpath: str) -> None:
    with open(trainconfigpath, "r") as f:
        trainconfig = yaml.safe_load(f)
    for key in [
        "train_h5",
        "batch_size",
        "learning_rate",
        "num_epochs",
        "noise_scales",
        "noise_centers",
        "gaussian_filter",
        "gaussian_sigma",
        "loss_functions",
        "modelfolder",
        "batch_size_inference",
        "evaluation_img_path",
        "evaluation_roi_folder",
        "stimulation_frames",
        "response_patience"
    ]:
        if key not in trainconfig:
            raise ValueError(
                f"Did not find required parameter {key} in {trainconfigpath}."
            )
    # create outputfolder
    modelfolder = os.path.abspath(trainconfig["modelfolder"])
    if os.path.exists(modelfolder):
        print(
            f"WARNING! Outputfolder ('{modelfolder}) for gridsearch already exists. Existing models might be overwritten"
        )
    os.makedirs(modelfolder, exist_ok=True)
    tmp_folder = os.path.join(modelfolder, "tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    # prepare paramterspace
    

    parameterspace = []
    # maybe suboptimal solution and smth like itertools should be used
    total_parameters = 0
    for ns in trainconfig["noise_scales"]:
        for nc in trainconfig["noise_centers"]:
            for lf in trainconfig["loss_functions"]:
                for gf in trainconfig["gaussian_filter"]:
                    if gf:
                        for sgf in trainconfig["gaussian_sigma"]:
                            parameterspace.append([ns, nc, gf, lf, sgf])
                            total_parameters += 1
                    else:
                        parameterspace.append([ns, nc, gf, lf, 0])
    print(
        f"Parameterspace contains {total_parameters} and thus has to train {len(parameterspace)} model(s)."
    )
    print('Evaluate on raw image')
    raw_result = raw_evaluate(trainconfig['evaluation_img_path'],
                              trainconfig['evaluation_roi_folder'],
                              trainconfig['stimulation_frames'],
                              trainconfig['response_patience'])
    with open(os.path.join(modelfolder,f'raw_performance.json'),'w') as outfile:
        json.dump(raw_result,outfile)
    with alive_bar(len(parameterspace)) as bar:
        for params in parameterspace:
            ns, nc, gf, lf, sgf = params
            modelname = f"unet_{lf}-loss_noisescale-{ns}_noisecenter-{nc}_gaussian-{gf}_sigma-{sgf}.pt"
            modelpath = os.path.join(modelfolder, modelname)
            history_savepath = os.path.join(
                modelfolder,
                f"unet_{lf}-loss_noisescale-{ns}_noisecenter-{nc}_gaussian-{gf}_sigma-{sgf}.npy",
            )
            dataloader = DataLoader(
                trainconfig["train_h5"],
                trainconfig["batch_size"],
                noise_center=nc,
                noise_scale=ns,
                apply_gausian_filter=gf,
                sigma_gausian_filter=sgf,
            )
            # train a model with the given parameters
            model = UNet(1)
            train(
                model,
                dataloader,
                trainconfig["num_epochs"],
                trainconfig["learning_rate"],
                lf,
                modelpath,
                history_savepath,
                False
            )
            model.to('cpu')
            del model
            # evaluate
            results_model = evaluate(modelpath,
                     tmp_folder,
                     trainconfig['evaluation_img_path'],
                     trainconfig['batch_size_inference'],
                     trainconfig['evaluation_roi_folder'],
                     trainconfig['stimulation_frames'],
                     trainconfig['response_patience'],
                     raw_result)
            with open(os.path.join(modelfolder,f'unet_{lf}-loss_noisescale-{ns}_noisecenter-{nc}_gaussian-{gf}_sigma-{sgf}_performance.json'),'w') as outfile:
                json.dump(results_model,outfile)
            bar()
