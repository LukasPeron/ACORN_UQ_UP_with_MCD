import click
from . import train_stage, infer_stage, eval_stage, calib_stage


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file")
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
@click.option("--sweep", "-s", default=None, help="Sweep parameter for training")
@click.option(
    "--checkpoint_resume_dir",
    "-r",
    default=None,
    help="Directory to resume checkpoint from",
)
@click.option(
    "--load_only_model_parameters",
    default=False,
    type=bool,
    help="Load only model parameters from checkpoint instead of the full training states",
)
def train(
    config_file, checkpoint, sweep, checkpoint_resume_dir, load_only_model_parameters
):
    train_stage.main(
        config_file,
        checkpoint,
        sweep,
        checkpoint_resume_dir,
        load_only_model_parameters,
    )


@cli.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for inference"
)
def infer(config_file, verbose, checkpoint):
    infer_stage.main(config_file, verbose, checkpoint)


@cli.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
@click.option("--dataset", "-d", default=None, help="Dataset to use for evaluation")
def eval(config_file, verbose, checkpoint, dataset):
    eval_stage.main(config_file, verbose, checkpoint, dataset)

@cli.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
@click.option("--dataset", "-d", default=None, help="Dataset to use for evaluation")
def calib(config_file, verbose, checkpoint, dataset):
    calib_stage.main(config_file, verbose, checkpoint, dataset)

if __name__ == "__main__":
    cli()
