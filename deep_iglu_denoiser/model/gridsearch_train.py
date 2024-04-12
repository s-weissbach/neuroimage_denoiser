from deep_iglu_denoiser.model.train import train
from deep_iglu_denoiser.model.unet import UNet
from deep_iglu_denoiser.utils.dataloader import DataLoader

import os
import torch.nn as nn
from alive_progress import alive_bar
import yaml


def gridsearch_train(trainconfigpath: str, keep_best_weights: bool = True) -> None:
    # create outputfolder
    modelfolder = os.absfolder(modelfolder)
    if os.path.exists(modelfolder):
        print(
            f"WARNING! Outputfolder ('{modelfolder}) for gridsearch already exists. Existing models might be overwritten"
        )
    os.makedirs(modelfolder, exist_ok=True)
    # prepare paramterspace
    with open(trainconfigpath, "r") as f:
        trainconfig = yaml.safe_load(f)
    for key in [
        "train_h5",
        "batch_size",
        "learning_rate",
        "num_epchs",
        "noisescales",
        "noisecenters",
        "gaussian_filter",
        "sigma_gaussian_filter",
        "loss_functions",
        "modelfolder",
    ]:
        if key not in trainconfig:
            raise ValueError(
                f"Did not find required parameter {key} in {trainconfigpath}."
            )

    parameterspace = []
    # maybe suboptimal solution and smth like itertools should be used
    total_parameters = 0
    for ns in trainconfig["noisescale"]:
        for nc in trainconfig["noisecenters"]:
            for lf in trainconfig["loss_functions"]:
                for gf in trainconfig["gaussian_filter"]:
                    if gf:
                        for sgf in trainconfig["sigma_gaussian_filter"]:
                            parameterspace.append([ns, nc, gf, lf, sgf])
                            total_parameters += 1
                    else:
                        parameterspace.append([ns, nc, gf, lf, 0])
    print(
        f"Parameterspace contains {total_parameters} and thus has to train {len(parameterspace)} model(s)."
    )

    model_overview = []
    with alive_bar(len(parameterspace)) as bar:
        for ns, nc, gf, lf, sgf in parameterspace:
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
            )
            bar()
