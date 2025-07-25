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

import importlib
import os
import warnings
from itertools import product
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from class_resolver import ClassResolver

from acorn.stages.track_building.utils import rearrange_by_distance
from acorn.utils import eval_utils
from acorn.utils.version_utils import get_pyg_data_keys
from acorn.utils.loading_utils import (
    add_variable_name_prefix_in_pyg,
    remove_variable_name_prefix_in_pyg,
    infer_num_nodes,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from acorn.utils import (
    load_datafiles_in_dir,
    run_data_tests,
    handle_weighting,
    handle_hard_cuts,
    remap_from_mask,
    handle_edge_features,
    get_optimizers,
    get_condition_lambda,
    get_variable_type,
    VariableType,
)
from acorn.stages.graph_construction.models.utils import graph_intersection

# TODO: What is this for??
torch.multiprocessing.set_sharing_strategy("file_system")


class EdgeClassifierStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.save_hyperparameters(hparams)

        # Assign hyperparameters
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_resolver = ClassResolver(
            [
                GraphDataset,
            ],
            base=Dataset,
            default=GraphDataset,
        )

    def setup(self, stage="fit", input_dir="stage_dir"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """
        if stage in ["fit", "predict"]:
            preprocess = True
            input_dir = "input_dir"
            self.load_data(stage, self.hparams[input_dir], preprocess)
            self.test_data(stage)
            torch.set_float32_matmul_precision(
                "medium" if stage == "fit" else "highest"
            )
        elif stage == "test":
            # during test stage, allow the possibility of
            if not self.hparams.get("reprocess_classifier"):
                if input_dir == "stage_dir":
                    preprocess = False
                elif input_dir == "input_dir":
                    preprocess = True
                print(f"Reading data from {input_dir} {'with' if preprocess else 'without'} preprocessing")
            self.load_data(stage, self.hparams[input_dir], preprocess)
            torch.set_float32_matmul_precision("highest")
        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("auc", summary="max")
        except Exception:
            warnings.warn(
                "Failed to define figures of merit, due to logger unavailable"
            )

    def load_data(self, stage, input_dir, preprocess=True):
        """
        Load in the data for training, validation and testing.
        """
        transform = None
        if self.hparams.get("transform") is not None:
            t_list = []
            for t in self.hparams["transform"]:
                # dynamically initiate transformations from pyg. The 'transform' configuration should be a list of elements like
                # {module_name: torch_geometric.transforms, class_name: ToSparseTensor, init_kwargs: {remove_edge_index: false}}
                module = importlib.import_module(t["module_name"])
                t = getattr(module, t["class_name"])(**t["init_kwargs"])
                t_list.append(t)
            transform = T.Compose(t_list)
        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.hparams["data_split"]
        ):
            if data_num > 0:
                dataset = self.dataset_resolver.make(
                    self.hparams.get("dataset_class"),
                    input_dir=input_dir,
                    data_name=data_name,
                    num_events=data_num,
                    stage=stage,
                    hparams=self.hparams,
                    preprocess=preprocess,
                    transform=transform,
                )
                setattr(self, data_name, dataset)

    def test_data(self, stage):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = [
            "hit_x",
            "edge_index",
            "track_edges",
            "track_to_edge_map",
            "edge_y",
        ]
        optional_features = [
            "track_particle_id",
            "track_particle_nhits",
            "track_particle_primary",
            "track_particle_pdgId",
            "hit_region",
            "track_particle_pt",
        ]
        run_data_tests(
            [
                dataset
                for dataset in [self.trainset, self.valset, self.testset]
                if dataset is not None
            ],
            required_features,
            optional_features,
        )

    def train_dataloader(self):
        """
        Load the training set.
        """
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        return DataLoader(self.trainset, batch_size=1, num_workers=num_workers)

    def val_dataloader(self):
        """
        Load the validation set.
        """
        if self.valset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[1]
        return DataLoader(self.valset, batch_size=1, num_workers=num_workers)

    def test_dataloader(self):
        """
        Load the test set.
        """
        if self.testset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[2]
        return DataLoader(self.testset, batch_size=1, num_workers=num_workers)

    def predict_dataloader(self):
        """
        Load the prediction sets (which is a list of the three datasets)
        """
        dataloaders = [
            self.train_dataloader(),
            self.val_dataloader(),
            self.test_dataloader(),
        ]
        dataloaders = [
            dataloader for dataloader in dataloaders if dataloader is not None
        ]
        return dataloaders

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        max_training_graph_size = self.hparams.get("max_training_graph_size", None)
        if (
            max_training_graph_size is not None
            and batch.edge_index.shape[1] > max_training_graph_size
        ):
            return None
        output = self(batch)
        loss, pos_loss, neg_loss = self.loss_function(output, batch)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_pos_loss",
            pos_loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_neg_loss",
            neg_loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        return loss

    def loss_function(self, output, batch, balance="proportional"):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled
        with the `weighting` config option.
        """

        assert hasattr(batch, "edge_y"), (
            "The batch does not have a truth label. Please ensure the batch has a `y`"
            " attribute."
        )
        assert hasattr(batch, "edge_weights"), (
            "The batch does not have a weighting label. Please ensure the batch"
            " weighting is handled in preprocessing."
        )

        if balance not in ["equal", "proportional"]:
            warnings.warn(
                f"{balance} is not a proper choice for the loss balance. Use either 'equal' or 'proportional'. Automatically switching to 'proportional' instead."
            )
            balance = "proportional"

        negative_mask = ((batch.edge_y == 0) & (batch.edge_weights != 0)) | (
            batch.edge_weights < 0
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask],
            torch.zeros_like(output[negative_mask]),
            weight=batch.edge_weights[negative_mask].abs(),
            reduction="sum",
        )

        positive_mask = (batch.edge_y == 1) & (batch.edge_weights > 0)
        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask],
            torch.ones_like(output[positive_mask]),
            weight=batch.edge_weights[positive_mask].abs(),
            reduction="sum",
        )

        if balance == "proportional":
            n = positive_mask.sum() + negative_mask.sum()
            return (
                (positive_loss + negative_loss) / n,
                positive_loss.detach() / n,
                negative_loss.detach() / n,
            )
        else:
            n_pos, n_neg = positive_mask.sum(), negative_mask.sum()
            n = n_pos + n_neg
            return (
                positive_loss / n_pos + negative_loss / n_neg,
                positive_loss.detach() / n,
                negative_loss.detach() / n,
            )

    def shared_evaluation(self, batch, batch_idx):
        output = self(batch)
        loss, pos_loss, neg_loss = self.loss_function(output, batch)

        scores = torch.sigmoid(output)
        if hasattr(self, "calibrated") and self.calibrated:
            # Apply spline rescaling
            rescaling_spline = self.rescaling_parameters
            scores_np = scores.detach().cpu().numpy()
            scores_rescaled = rescaling_spline(scores_np)
            # Ensure scores remain in [0, 1] range
            scores_rescaled = np.clip(scores_rescaled, 0, 1)
            scores = torch.from_numpy(scores_rescaled).to(scores.device)
        batch.edge_scores = scores.detach()

        all_truth = batch.edge_y.bool()
        target_truth = (batch.edge_weights > 0) & all_truth

        return {
            "loss": loss,
            "all_truth": all_truth,
            "target_truth": target_truth,
            "output": output,
            "batch": batch,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
        }

    def validation_step(self, batch, batch_idx):
        output_dict = self.shared_evaluation(batch, batch_idx)
        self.log_metrics(
            output_dict["output"],
            output_dict["all_truth"],
            output_dict["target_truth"],
            output_dict["loss"],
        )
        self.log(
            "val_loss",
            output_dict["loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "val_pos_loss",
            output_dict["pos_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "val_neg_loss",
            output_dict["neg_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        return self.shared_evaluation(batch, batch_idx)

    def log_metrics(self, output, all_truth, target_truth, loss):
        scores = torch.sigmoid(output)
        preds = scores > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        target_true = target_truth.sum().float()
        target_true_positive = (target_truth.bool() & preds).sum().float()
        all_true_positive = (all_truth.bool() & preds).sum().float()

        total_auc = roc_auc_score(
            all_truth.cpu().detach(),
            scores.float().cpu().detach(),
        )

        truth_without_nontarget = all_truth[(target_truth) | (~all_truth)]
        scores_without_nontarget = scores[(target_truth) | (~all_truth)]
        # add torch.sigmoid(output).float() to convert to float in case training is done with 16-bit precision
        auc = roc_auc_score(
            truth_without_nontarget.cpu().detach(),
            scores_without_nontarget.float().cpu().detach(),
        )
        true_and_fake_positive = (
            edge_positive - (preds & (~target_truth) & all_truth).sum().float()
        )

        target_eff = target_true_positive / target_true
        target_pur = target_true_positive / edge_positive
        total_pur = all_true_positive / edge_positive
        purity = target_true_positive / true_and_fake_positive
        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log_dict(
            {
                "current_lr": current_lr,
                "eff": target_eff,
                "target_pur": target_pur,
                "total_pur": total_pur,
                "pur": purity,
                "auc": auc,
                "total_auc": total_auc,
            },  # type: ignore
            sync_dist=True,
            batch_size=1,
            on_epoch=True,
            on_step=False,
        )

        return preds

    def on_train_epoch_start(self):
        self.trainer.strategy.optimizers = [
            self.trainer.lr_scheduler_configs[0].scheduler.optimizer
        ]

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], self.hparams.get("min_lr", 0))

        if self.hparams.get("debug") and self.trainer.current_epoch == 0:
            warnings.warn("DEBUG mode is on. Will print out gradient if encounter None")
            invalid_gradient = False
            for param in self.parameters():
                if param.grad is None:
                    warnings.warn(
                        "Some parameters get non-numerical gradient. Check model and"
                        " train settings"
                    )
                    invalid_gradient = True
                    break
            if invalid_gradient:
                print([param.grad for param in self.parameters()])
            self.hparams["debug"] = False

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an edge-scoring model on the input graph
        2. Add the scored edges to the graph, as `scores` attribute
        3. Append the stage config to the `config` attribute of the graph
        """
        dataset = self.predict_dataloader()[dataloader_idx].dataset
        event_id = (
            batch.event_id[0] if isinstance(batch.event_id, list) else batch.event_id
        )
        if os.path.exists(
            os.path.join(
                self.hparams["stage_dir"],
                dataset.data_name,
                f"event{event_id}.pyg",
            )
        ) and self.hparams.get("skip_existing"):
            return
        if batch.edge_index.shape[1] == 0:
            batch.edge_scores = torch.tensor([], device=self.device)
        else:
            eval_dict = self.shared_evaluation(batch, batch_idx)
            batch = eval_dict["batch"]
        self.save_edge_scores(batch, dataset)

    def save_edge_scores(self, event, dataset):
        event = dataset.unscale_features(event)

        event.config.append(self.hparams)
 
        event.edge_index = rearrange_by_distance(event, event.edge_index)  
        event.track_edges = rearrange_by_distance(event, event.track_edges)
 
        event.track_to_edge_map = graph_intersection(
            event.edge_index,
            event.track_edges,
            return_y_pred=False,
            return_y_truth=False,
            return_truth_to_pred=True,
        )

        datatype = dataset.data_name
        os.makedirs(os.path.join(self.hparams["stage_dir"], datatype), exist_ok=True)
        event_id = (
            event.event_id[0] if isinstance(event.event_id, list) else event.event_id
        )
        if not self.hparams.get("variable_with_prefix"):
            event = remove_variable_name_prefix_in_pyg(event)
        torch.save(
            event.cpu(),
            os.path.join(self.hparams["stage_dir"], datatype, f"event{event_id}.pyg"),
        )

    @classmethod
    def evaluate(cls, config, checkpoint=None):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config).to(device)
        if checkpoint is not None:
            print(f"Restoring model from {checkpoint}")
            graph_constructor = cls.load_from_checkpoint(checkpoint, hparams=config).to(
                device
            )

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            # If we are doing Uncertainty Quantification with MCDropout we are running inference of the model on its input graphs. In the other cases (evaluation of eff, pur, ROC, etc.) we want to use the output graphs
            if plot_function == "graph_mcdropout_uncertainty":
                graph_constructor.setup(stage="test", input_dir="input_dir")
            else:
                graph_constructor.setup(stage="test", input_dir="stage_dir")
                graph_constructor.eval()
            if hasattr(eval_utils, plot_function):
                getattr(eval_utils, plot_function)(
                    graph_constructor, plot_config, config
                )
            else:
                print(f"Plot {plot_function} not implemented")

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.edge_scores >= score_cut

        # flip edge direction if points inward
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)

        event.track_to_edge_map = graph_intersection(
            event.edge_index,
            event.track_edges,
            return_y_pred=False,
            return_y_truth=False,
            return_truth_to_pred=True,
        )
        event.track_to_passing_edge_map = graph_intersection(
            event.edge_index[:, passing_edges_mask],
            event.track_edges,
            return_y_pred=False,
            return_truth_to_pred=True,
        )
        event.edge_pred = passing_edges_mask

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(
            event.track_to_passing_edge_map.shape[0], dtype=torch.bool
        ).to(self.device)

        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            passing_tracks = passing_tracks.to(self.device) * condition_lambda(event)

        event.track_target_mask = passing_tracks

    @staticmethod
    def MCD_fwd_pass(lightning_module, dataset, n_mcd_passes):
        # Use only for the calibration stage, different from the evaluation stage
        # Do not make the distinction between target and non-target edges
        from tqdm import tqdm

        n_event = len(dataset)
        all_truth = [None for _ in range(len(dataset))]     # Store truth once per event
        all_scores = [[] for _ in range(n_mcd_passes)]

        for i in tqdm(range(n_mcd_passes)):
            for num_event, event in enumerate(dataset): #* same code as in graph_region_efficiency_purity()
                with torch.inference_mode():
                    lightning_module.train()
                    eval_dict = lightning_module.shared_evaluation(
                        event.to(lightning_module.device), 0
                    )
                event = eval_dict["batch"]
                edge_scores = event.edge_scores.cpu().numpy()
                all_scores[i].append(edge_scores)
                # Handling of true edges
                if all_truth[num_event] is None:
                    all_truth[num_event] = event.edge_y.cpu().numpy()
        all_mean_scores = [[] for _ in range(len(dataset))]
        for n_event in range(len(dataset)):
            temp_all_scores = []
            for i in range(n_mcd_passes):
                temp_all_scores.append(all_scores[i][n_event])
            all_mean_scores[n_event] = np.mean(temp_all_scores, axis=0)
        # Flatten arrays for analysis
        all_flat_scores = np.concatenate(all_mean_scores)
        all_flat_truth = np.concatenate(all_truth)
        if hasattr(lightning_module, "rescaling_parameters"):
            print("Rescaling of scores")
            rescaling_spline = lightning_module.rescaling_parameters
            all_flat_scores = rescaling_spline(all_flat_scores)
            # Ensure scores remain in [0, 1] range
            all_flat_scores = np.clip(all_flat_scores, 0, 1)
        return all_flat_scores, all_flat_truth

    @staticmethod
    def reliability(all_flat_scores, all_flat_truth):
        """
        Calculate the rescaling parameters for the reliability of the scores using adaptive binning.
        """
        from scipy.interpolate import UnivariateSpline
        import numpy as np

        # Create adaptive bins with higher resolution in 0-0.2 region
        low_bins = np.linspace(0, 0.2, 1001)  # High resolution for 0-0.2
        high_bins = np.linspace(0.2, 1.0, 201)  # Normal resolution for 0.2-1.0
        score_bins = np.concatenate([low_bins, high_bins[1:]])  # Remove duplicate 0.2
        
        bin_indices = np.digitize(all_flat_scores, score_bins) - 1

        bin_centers = []
        reliability = []

        for i in range(len(score_bins) - 1):
            mask = bin_indices == i
            bin_size = np.sum(mask)
            
            # Use smaller minimum bin size for low score region
            min_bin_size = 3 if score_bins[i] <= 0.2 else 10
            
            if bin_size > min_bin_size:
                bin_truth = all_flat_truth[mask]
                true_count = np.sum(bin_truth)
                bin_reliability = true_count / bin_size if bin_size > 0 else 0
                bin_center = (score_bins[i] + score_bins[i + 1]) / 2
                bin_centers.append(bin_center)
                reliability.append(bin_reliability)
        
        bin_centers = np.array(bin_centers)
        reliability = np.array(reliability)
        
        k = min(5, len(bin_centers) - 1)
        s = 0  # No smoothing for exact fit
        spline = UnivariateSpline(bin_centers, reliability, k=k, s=s)
        
        return spline

    @classmethod
    def calibrate(cls, config, checkpoint=None):
        """
        The gateway for the calibration stage. This class method is called from the calib_stage.py script.
        """
        import tqdm
        print("checkpoint:", checkpoint)
        # Load data from testset directory
        graph_constructor = cls(config).to(device)
        if checkpoint is not None:
            print(f"Restoring model from {checkpoint}")
            graph_constructor = cls.load_from_checkpoint(checkpoint, hparams=config).to(
                device
            )
        graph_constructor.setup(stage="test", input_dir="input_dir")
        dataset_name = config["dataset"]
        dropout_value = config.get("hidden_dropout", 0.0)
        dataset = getattr(graph_constructor, dataset_name)
        dropout_str = f"_dropout{dropout_value:.1f}".replace('.', 'p')
        # Load valset and run inference without mcdropout, keep scores and truth
        scores = []
        truth_array = []
        for num_event, event in tqdm.tqdm(enumerate(dataset)):
            with torch.inference_mode():
                graph_constructor.eval()
                eval_dict = graph_constructor.shared_evaluation(
                    event.to(graph_constructor.device), 0
                )
            event = eval_dict["batch"]
            edge_scores = event.edge_scores.cpu().numpy()
            scores.append(edge_scores)
            edge_truth = event.edge_y.cpu().numpy()
            truth_array.append(edge_truth)
        all_flat_scores = np.concatenate(scores)
        all_flat_truth = np.concatenate(truth_array)
        rescaling_parameters = EdgeClassifierStage.reliability(all_flat_scores, all_flat_truth)
        graph_constructor.rescaling_parameters = rescaling_parameters
        graph_constructor.calibrated = True
        scores_scaled = []
        truth_array_scaled = []

        for num_event, event in tqdm.tqdm(enumerate(dataset)):
            with torch.inference_mode():
                graph_constructor.eval()
                eval_dict = graph_constructor.shared_evaluation(
                    event.to(graph_constructor.device), 0
                )
            event = eval_dict["batch"]
            edge_scores = event.edge_scores.cpu().numpy()
            scores_scaled.append(edge_scores)
            edge_truth = event.edge_y.cpu().numpy()
            truth_array_scaled.append(edge_truth)
        all_flat_scores_scaled = np.concatenate(scores_scaled)
        all_flat_truth = np.concatenate(truth_array_scaled)

        # Save spline object
        import pickle
        print(config["stage_dir"])
        os.makedirs(os.path.join(config["stage_dir"], "calibration"), exist_ok=True)
        with open(os.path.join(config["stage_dir"], "calibration/rescaling_spline.pkl"), 'wb') as f:
            pickle.dump(rescaling_parameters, f)
        config["dataset_size"] = len(dataset)
        from acorn.utils.eval_utils import plot_calibration_curve
        plot_calibration_curve(all_flat_scores_scaled, all_flat_truth, dataset, config, plot_config="", dropout_str=dropout_str, dropout_value=dropout_value, from_calibration_stage=True)
        from acorn.utils.eval_utils import plot_reliability_diagram
        plot_reliability_diagram(all_flat_scores_scaled, all_flat_truth, dataset, config, plot_config="", dropout_str=dropout_str, dropout_value=dropout_value, from_calibration_stage=True)


class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preprocess=True,
    ):
        if hparams is None:
            hparams = {}
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage
        self.preprocess = preprocess
        self.transform = transform

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility

    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        # convert DataBatch to Data instance because some transformations don't work on DataBatch
        event = Data(**event.to_dict())
        if (not self.hparams.get("variable_with_prefix")) or self.hparams.get(
            "add_variable_name_prefix_in_pyg"
        ):
            event = add_variable_name_prefix_in_pyg(event)
        if not self.preprocess:
            return event
        event = self.preprocess_event(event)
        # do pyg transformation if a torch_geometric.transforms instance is given
        if self.transform is not None:
            event = self.transform(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        infer_num_nodes(event)
        event = self.apply_hard_cuts(event)
        event = self.construct_weighting(event)
        event = self.handle_edge_list(event)
        event = self.scale_features(event)
        if self.hparams.get("edge_features") is not None:
            event = self.add_edge_features(
                event
            )  # scaling must be done before adding features
        return event

    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if (
            self.hparams is not None
            and "hard_cuts" in self.hparams.keys()
            and self.hparams["hard_cuts"]
        ):
            assert isinstance(
                self.hparams["hard_cuts"], dict
            ), "Hard cuts must be a dictionary"
            handle_hard_cuts(event, self.hparams["hard_cuts"])

        return event

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """

        assert event.edge_y.shape[0] == event.edge_index.shape[1], (
            f"Input graph has {event.edge_index.shape[1]} edges, but"
            f" {event.edge_y.shape[0]} truth labels"
        )

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(
                self.hparams["weighting"][0], dict
            ), "Weighting must be a list of dictionaries"
            event.edge_weights = handle_weighting(event, self.hparams["weighting"])
        else:
            event.edge_weights = torch.ones_like(event.edge_y, dtype=torch.float32)

        return event

    def handle_edge_list(self, event):
        if (
            "input_cut" in self.hparams.keys()
            and self.hparams["input_cut"] # ensure input_cut is not 0.0
            and "edge_scores" in get_pyg_data_keys(event)
        ):
            # Apply a score cut to the event
            # print("I DO APPLY INPUT CUT")
            self.apply_score_cut(event, self.hparams["input_cut"])

        # if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
        #     # Flip event.edge_index and concat together
        #     self.to_undirected(event)
        return event

    def to_undirected(self, event):
        """
        Add the reverse of the edge_index to the event. This then requires all edge features to be duplicated.
        Additionally, the truth map must be duplicated.
        """
        num_edges = event.edge_index.shape[1]
        # Flip event.edge_index and concat together
        event.edge_index = torch.cat(
            [event.edge_index, event.edge_index.flip(0)], dim=1
        )
        # event.edge_index, unique_edge_indices = torch.unique(event.edge_index, dim=1, return_inverse=True)
        event.track_edges = torch.cat(
            [event.track_edges, event.track_edges.flip(0)], dim=1
        )

        # Concat all edge-like and track-like features together
        for key in get_pyg_data_keys(event):
            if key in {"track_to_edge_map", "edge_index", "track_edges"}:
                continue
            if not isinstance(event[key], torch.Tensor) or not event[key].shape:
                continue
            if get_variable_type(key) in {
                VariableType.EDGE_LIKE,
                VariableType.TRACK_LIKE,
            }:
                event[key] = torch.cat([event[key], event[key]], dim=0)

        # handle truth_map separately
        track_to_edge_map = event.track_to_edge_map.clone()
        track_to_edge_map[track_to_edge_map >= 0] = (
            track_to_edge_map[track_to_edge_map >= 0] + num_edges
        )
        event.track_to_edge_map = torch.cat(
            [event.track_to_edge_map, track_to_edge_map], dim=0
        )

        return event

    def add_edge_features(self, event):
        if "edge_features" in self.hparams.keys():
            assert isinstance(
                self.hparams["edge_features"], list
            ), "Edge features must be a list of strings"
            handle_edge_features(event, self.hparams["edge_features"])
        return event

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in get_pyg_data_keys(
                    event
                ), f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

        return event

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in get_pyg_data_keys(
                    event
                ), f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]
        return event

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.edge_scores >= score_cut
        for key in get_pyg_data_keys(event):
            if (
                isinstance(event[key], torch.Tensor)
                and get_variable_type(key) == VariableType.EDGE_LIKE
            ):
                event[key] = event[key][..., passing_edges_mask]

        remap_from_mask(event, passing_edges_mask)
        return event

    def get_y_node(self, event):
        y_node = torch.zeros(event.z.size(0))
        y_node[event.track_edges.view(-1)] = 1
        event.y_node = y_node
        return event


class HeteroGraphMixin:
    def __init__(self) -> None:
        self.hparams = {}

    def get_node_type(self, event):
        assert "region_ids" in self.hparams.keys() and isinstance(
            self.hparams["region_ids"], list
        ), "To create a heterogeneous graph, must define region id"
        region = event.region
        node_type = torch.zeros_like(region, dtype=torch.int64)
        node_type_name = []

        for idx, region_id in enumerate(self.hparams["region_ids"]):
            mask = torch.isin(region, torch.tensor(region_id["id"]))
            node_type[mask] = idx
            node_type_name.append(region_id["name"])

        event.node_type = node_type
        event.node_type_name = node_type_name
        return event

    def get_edge_type(self, event):
        assert (
            "node_type" in event
        ), "event must have node_type. Run it through self.get_node_type"
        assert "region_ids" in self.hparams.keys() and isinstance(
            self.hparams["region_ids"], list
        ), "To create a heterogeneous graph, must define region id"
        edge_index = event.edge_index
        node_type = event.node_type
        edge_type = torch.zeros_like(edge_index[0], dtype=torch.int64)
        edge_type_name = []

        for id, link in enumerate(product(node_type.unique(), node_type.unique())):
            src, dst = link
            mask = (node_type[edge_index[0]] == src) & (node_type[edge_index[1]] == dst)
            edge_type[mask] = id
            edge_type_name.append(
                (
                    self.hparams["region_ids"][src]["name"],
                    "to",
                    self.hparams["region_ids"][dst]["name"],
                )
            )

        event.edge_type = edge_type
        event.edge_type_name = edge_type_name
        return event

    def convert_heterogeneous(self, event):
        assert (
            "node_type" in event
            and "edge_type" in event
            and "node_type_name" in event
            and "edge_type_name" in event
        ), "Must run through both self.get_node_type and self.get_edge_type"
        return event.to_heterogeneous(
            node_type=event.node_type,
            edge_type=event.edge_type,
            node_type_names=event.node_type_name,
            edge_type_names=event.edge_type_name,
        )

    def get_input_data(self, event):
        event.input_node_features = torch.stack(
            [event[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()
        edge_feature_list = self.hparams.get("edge_features", [])
        if len(edge_feature_list) > 0:
            event.input_edge_features = torch.stack(
                [event[feature] for feature in edge_feature_list], dim=-1
            ).float()
        return event


class HeteroGraphDataset(GraphDataset, HeteroGraphMixin):
    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams={},
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preprocess=True,
    ):
        super().__init__(
            input_dir,
            data_name,
            num_events,
            stage,
            hparams,
            transform,
            pre_transform,
            pre_filter,
            preprocess,
        )

    def preprocess_event(self, event):
        if self.hparams.get("undirected"):
            event = self.to_undirected(event)
        infer_num_nodes(event)
        event = self.apply_hard_cuts(event)
        event = self.construct_weighting(event)
        event = self.handle_edge_list(event)
        event = self.add_edge_features(event)
        event = self.scale_features(event)
        event = Data(**event.to_dict())
        event = self.get_input_data(event)
        event = self.get_node_type(event)
        event = self.get_edge_type(event)
        event = self.convert_heterogeneous(event)
        return event


class DirectedHeteroGraphDataset(GraphDataset, HeteroGraphMixin):
    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams={},
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preprocess=True,
    ):
        super().__init__(
            input_dir,
            data_name,
            num_events,
            stage,
            hparams,
            transform,
            pre_transform,
            pre_filter,
            preprocess,
        )

    def handle_direction(self, event):
        # get the distance squared
        r2 = event.r**2 + event.z**2

        inward_edge_mask = r2[event.edge_index[0]] > r2[event.edge_index[1]]
        inward_track_edge_mask = r2[event.track_edges[0]] > r2[event.track_edges[1]]

        event.edge_index[:, inward_edge_mask] = event.edge_index[
            :, inward_edge_mask
        ].flip(0)
        event.track_edges[:, inward_track_edge_mask] = event.track_edges[
            :, inward_track_edge_mask
        ].flip(0)

        return event

    def preprocess_event(self, event):
        self.hparams["undirected"] = False
        event = super().preprocess_event(event)
        event = Data(**event.to_dict())
        event = self.handle_direction(event)
        event = self.get_input_data(event)
        event = self.get_node_type(event)
        event = self.get_edge_type(event)
        event = self.convert_heterogeneous(event)
        return event


class HeteroGraphDatasetWithNode(GraphDataset, HeteroGraphMixin):
    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams={},
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preprocess=True,
    ):
        super().__init__(
            input_dir,
            data_name,
            num_events,
            stage,
            hparams,
            transform,
            pre_transform,
            pre_filter,
            preprocess,
        )

    def preprocess_event(self, event):
        event = super().preprocess_event(event)
        event = Data(**event.to_dict())
        event = self.get_y_node(event)
        event = self.get_input_data(event)
        event = self.get_node_type(event)
        event = self.get_edge_type(event)
        event = self.convert_heterogeneous(event)
        return event
