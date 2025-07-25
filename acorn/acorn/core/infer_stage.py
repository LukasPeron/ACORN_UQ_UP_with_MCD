# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script:
1. Loads a training config
2. Checks the stage to train
3. Loads the stage module
4. Trains the stage
5. Tests the output
"""


import sys

import yaml
import click
import logging
import torch
import os

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule

from .core_utils import str_to_class, find_latest_checkpoint
from acorn.utils.loading_utils import add_variable_name_prefix_in_config


@click.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
def main(config_file, verbose, checkpoint):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    infer(config_file, verbose, checkpoint)


def infer(config_file, verbose=None, checkpoint=None):
    # set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not config.get("variable_with_prefix"):
        config = add_variable_name_prefix_in_config(config)

    # load stage
    stage = config["stage"]
    model = config["model"]

    stage_module = str_to_class(stage, model)

    if issubclass(stage_module, LightningModule):
        lightning_infer(config, stage_module, checkpoint)
    else:
        stage_module.infer(config)


def lightning_infer(config, stage_module, checkpoint=None):
    checkpoint_path = (
        checkpoint
        if checkpoint
        else find_latest_checkpoint(
            config["stage_dir"], templates=["best*.ckpt", "*.ckpt"]
        )
    )
    if not checkpoint_path:
        print("No checkpoint found")
        sys.exit(1)
    print(f"Loading checkpoint: {checkpoint_path}")

    stage_module = stage_module.load_from_checkpoint(
        checkpoint_path, map_location=torch.device("cpu")
    )
    stage_module._hparams = {**stage_module._hparams, **config}
    if (not config.get("variable_with_prefix")) or config.get(
        "add_variable_name_prefix_in_ckpt"
    ):
        stage_module._hparams = add_variable_name_prefix_in_config(
            stage_module._hparams
        )

    # setup stage
    stage_module.setup(stage="predict")
    stage_module.calibrated = config.get("calibration", False)
    if stage_module.calibrated:
        print("Using calibrated model")
        # Load spline object instead of text file
        import pickle
        spline_path = os.path.join(config["stage_dir"], "calibration", "rescaling_spline.pkl")
        with open(spline_path, 'rb') as f:
            rescaling_parameters = pickle.load(f)
        stage_module.rescaling_parameters = rescaling_parameters
        print(f"Loaded spline rescaling parameters from {spline_path}")
        
    accelerator = config.get("accelerator")
    devices = config.get("devices")

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
    )

    with torch.inference_mode():
        trainer.predict(stage_module)


if __name__ == "__main__":
    main()
