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
1. Loads a configuration file.
2. Load a trained model from a checkpoint.
3. Runs the calibration stage on the specified dataset (trainset, valset, or testset).
4. Logs the results.


The calibration stage consist on :
1. Do N Monte Carlo Dropout forward passes on the input data.
2. Compute the resulting reliability distribution.
3. Save the resulting distribution and provide edge_scores rescaling
4. Do a second N Monte Carlo Dropout forward passes on the input data.
5. Compute the resulting calibration curve
6. Save the resulting optimal edge_scores cut value
"""

import sys

import yaml
import click
import logging

from pytorch_lightning import LightningModule
import torch

from .core_utils import str_to_class, find_latest_checkpoint
from acorn.utils.loading_utils import add_variable_name_prefix_in_config


@click.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
@click.option(
    "--dataset",
    "-d",
    default="valset",
    type=click.Choice(["trainset", "valset", "testset"], case_sensitive=True),
)
def main(config_file, verbose, checkpoint, dataset):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """

    calibrate(config_file, verbose, checkpoint, dataset)


def calibrate(config_file, verbose=None, checkpoint=None, dataset="valset"):
    # set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module = str_to_class(stage, model)
    config["dataset"] = dataset

    if not issubclass(stage_module, LightningModule):
        raise ValueError(f"{stage} is not a valid LightningModule")

    checkpoint_path = (
        find_latest_checkpoint(
            config["stage_dir"], templates=["best*.ckpt", "*.ckpt"]
        )
        if checkpoint is None
        else checkpoint
    )
    if not checkpoint_path:
        print("No checkpoint found")
        sys.exit(1)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_config = torch.load(
        checkpoint_path, map_location=torch.device("cpu")
    )["hyper_parameters"]
    config = {**checkpoint_config, **config}

    if (not config.get("variable_with_prefix")) or config.get(
        "add_variable_name_prefix_in_ckpt"
    ):
        config = add_variable_name_prefix_in_config(config)

    stage_module.calibrate(config, checkpoint_path)

if __name__ == "__main__":
    main()