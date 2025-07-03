# Example TrackLM UQ : full ACORN pipeline training with `metric_learning`, `filter`, `interaction_gnn` and `cc_and_walk`.
#### Author : Lukas Péron (lukas.peron@ens.psl.eu), version of 05/27/2025
## What will you find in this example ?
This example folder is made for any new student or person interested in applying a full ACORN pipeline to a pattern recognition task. It contains 12 `.yaml` config files that can be used to train and use the ACORN pipeline on the full TrackML dataset (without $p_T$ hardcut). The rest of this readme explains how to run each part of the training, inference and evaluation of the pipeline. We will also present methods for Uncertainty Quantification and score calibration of the pipeline components. In each config file you will find a complete documentation.
## Setup
**CAUTION: Some parts of this code are made to be used with modified versions of the main branch codes. These parts are indicated bellow with a $\dagger$ and can be found at the end of this readme. 

<u>**TODO: list changes made from main branch and add new branch UQ_MCD to ACORN**</u>

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files. These are TrackML events pulled from CodaLab, as well as a detector description file. Define your data directory MY_DATA_DIR with
```bash
data_dir=MY_DATA_DIR
```
You can choose whether to download a small dataset of 100 events `trackml_100_events.zip` (appropriate for toy model training, around 700Mb) or a larger dataset of 1500 events `trackml_1500_events.zip` (appropriate for full event training, around 11Gb):
```bash
wget https://portal.nersc.gov/cfs/m3443/dtmurnane/TrackML_Example/trackml_XXX_events.zip -P $data_dir/Example_TrackML_UQ
wget https://portal.nersc.gov/cfs/m3443/dtmurnane/TrackML_Example/detectors.csv -P $data_dir/Example_TrackML_UQ
unzip $data_dir/Example_4/trackml_XXX_events.zip -d $data_dir/Example_4
```
The location of this data, as well as all parameters controlling the GNN4ITk reconstruction chain, is specified in `yaml` config files. The data directory currently has a placeholder MY_DATA_DIR. Replace this with the actual data directory with
```bash
sed -i "s/MY_DATA_DIR/$data_dir/g" *.yaml
```
# Running the Example
The following commands will run the acorn pipeline. In general, they follow the pattern
```
train --> infer --> eval
```
where `train` is used to train a model, `infer` is used to apply the model to data, and `eval` is used to evaluate the performance of the model. If a model has already been trained, we do not need to train it, only provide the model checkpoint file to the `infer` step.
## Building data
First, we build our input data from the raw Athena events:
```bash
acorn infer data_reader.yaml
```
## Metric Learning
We start the graph construction by training the Metric Learning stage:
```bash
acorn train metric_learning_train.yaml
```
Then, we build graphs using the Metric Learning in inference:
```bash
acorn infer metric_learning_infer.yaml
```
(Recommended) We can examine the performance of the Metric Learning, by printing some efficiency plots 


```bash
acorn eval metric_learning_eval.yaml
```

<u>**if time allows, here you could insert some sample plots so that folks know what a well-trained model should behave**:</u>
## Filter
If the graphs constructed from the metric learning stage are too large (they might be if, for instance, you are training on full events and the metric learning hasn't quite converged, or if you have a GPU with less than 10Gb memory), one can then apply a `Filter` step to reduce the number of edges in the graph. This is done by training a `Filter` model.
```bash
acorn train filter_train.yaml
```
Then, we apply the `Filter` in inference:
```bash
acorn infer filter_infer.yaml
```
And we can get some performance plots with:
```bash
acorn eval filter_eval.yaml
```
**NOTE: The filter step is not mandatory, however it is strongly recommanded if a realistic pipeline training is wanted.**
## Interaction GNN
Then, we train the GNN **(NOTE: if you trained the filter, then you should point the input directory to the filter stage directory, and comment out the "input_cut" configuration in the GNN training config file)**:
```bash
acorn train gnn_train.yaml
```
Once the GNN is trained, we apply the GNN in inference to produce a dataset of scored graphs:
```bash
acorn infer gnn_infer.yaml
```
Here again, one can plot some performance plots if needed:
```bash
acorn eval gnn_eval.yaml
```
## CC And Walk
Finally, we produce track candidates from the scored graphs:
```bash
acorn infer track_building_infer.yaml
```
And plot the performance
```bash
acorn eval track_building_eval.yaml
```
## Explanation of some hyper parameters used in the `.yaml` files
Some hyper parameters (hparams) used in the configuration files require a detailed explanation, including:
`weighting` : in each event, we aim to reconstruct some _target_ edges. In this example, those targets are edges that have at least 3 hits (ie. nodes) and correspond to particles track with $p_T > 1$ GeV. To enhance the training, each edge carries a _weight_ which is a multiplicative factor that will be applied in the loss computation. For instance, if an edge has weight 0 then it will have a null contribution to the loss computation. Usually, we want weights to have the following logic :
- non-target true edge (ie. edge with lower than 3 hits or $p_T < 1$ GeV) $\to$ weight $w = 0$
- fake edge $\to$ weight $w_f\neq0$
- target true edge $\to$ weight $w_t > w_f$. We want $w_t > w_f$ (e.g. $w_t=1$ and $w_f=0.1$) so that true edges are more important in the loss computation (and are then expected to be reconstructed more accurately).
`hard_cut` : this hparam allow to discard the data (hits) associated with certain particle tracks. For example, to evaluate a new method it is common to discard hits that correspond to particle track with $p_T < 1$ GeV.
<u>**CAUTION**</u> this it not to be mixed up with the weighting previously discussed. Hardcut actually reduces the graph size by removing nodes (and the corresponding edges) where the weighting only give some edges a null weight in the loss computation.


`target_tracks` : This feature specifies what will be considered "true target" in the `eval` stage. For instance, one may has trained the model with true target being all edges with $p_T > 1$ GeV but wants to evaluate its performances only on edges with $p_T > 5$ GeV.


`input_cut` : used in the `gnn_train` and `gnn_infer` configuration files. This hparam set an edge score cut on the input graph of the `InteractionGNN` step (which is the output graph of the `Filter` edge classification step). All edges with scores $s$ lower than the input_cut will be removed before being passed to the gnn. Reducing the input graph size, speeds up the training stage of the gnn and reduces the size of the output gnn graph.
## Uncertainty Quantification (UQ) and Calibration $\dagger$
### Monte Carlo Dropout (MCD) for UQ
This example uses [MCD][1] as an UQ method. MCD is off by default. We activate and control MCD by setting `input_dropout` and `hidden_dropout`, two (optional) configuration parameters of the training step config files:

- The `input_dropout` feature is the dropout probability after the input layer of either a classification MLP (for `metric_learning` and `filter`) or a message passing MLP (for `InteractionGNN`). We recommend to set it to 0 except when training the GNN. This is because the message passing MLP input is the gnn encoder output, ie. a high dimensionnal latent representation of the graph and not the graph itself. Setting `input_dropout` to 0 in `metric_learning` or `filter` implies to not pass the full graph in the classification MLP at each epoch.

- The `hidden_dropout` feature is the dropout probability after any hidden layer of either a classification MLP (for `metric_learning` and `filter`) or a message passing MLP (for `InteractionGNN`). Setting it to a non-zero value (we recommend 0.1) drops out (zeroes) a fraction of the hidden layer outputs.

The implementation of MCD is done in the `graph_mcdropout_uncertainty` function of `eval_utils.py` (located in `acorn/acorn/utils/eval_utils.py`). This method is called in the eval stage of any `edge_classifier` model (namely `filter` or `gnn`) as an argument to the plot functions. 

This method loads a PyTorch checkpoint (user-specified or retrieved automatically) in the PyTorch train mode in order to keep the dropout layers activated. It then proceed to infer the wanted dataset $N$ times (we recommend $N=100$). For a given graph, the $N$ stochastic inference will yield different edge scores as different weights of the model have been dropped. All of these scores are stocked along with other important features (eta and pt values, truth target, truth non target and false mask).

We then proceed to compute, for all edges of all graphs, the means and standard deviations of the stochastic scores. The standard deviations are then plotted against different metrics like eta or pt for all the different target/non target/false category.

We also propose a stronger analysis by considering an other way to quantify uncertainty by computing the mutual information of the scores for all edges in all graphs and the Shanon's entropy of these obtained mean scores (averaged on the stochastic passes). This procedure is detailed in section 3.3.1 of [the Yarin Gal's PhD thesis][2]. The mutual information correspond to `all_flat_epistemic_uncertainty` in the `graph_mcdropout_uncertainty` method and the Shanon's entropy to `all_flat_total_uncertainty`. The aleatoric uncertainty is defined then as `all_flat_total_uncertainty` - `all_flat_epistemic_uncertainty`. Here is an exhaustive list of all the plots that are produced :

- Uncertainty vs edge score
- Uncertainty vs $p_T$
- Uncertainty vs $\eta$
- Uncertainty histogram distribution
- Calibration curve
- Reliability curve
- Edge score distribution
- Number of edges vs $\eta$
- Edge score vs $\eta$
- Edge score vs $p_T$
- Uncertainty skewness
- Uncertainty kurtosis
- Aleatoric uncertainty
- Epistemic uncertainty (mutual information)
- Total uncertainty (Shanon's entropy)
- Uncertainty's entropy comparison with gaussian scores production

This last plot is an indicator of the non-gaussianity of the MCD scores predictions. It has an anagolous role as the skewness and kurtosis plot. However, since non gaussian curve can still have null skewness and kurtosis, we do this additionnal test. It is done as following: for each edge, we stochastically (ie. with MCD) infer $N$ scores. We compute the mean, standard deviation and entropy of this set of prediction. We then draw $N$ samples from a gaussian distribution with same mean and standard deviation, and we compute the entropy of the gaussian set of predictions. Then, we compare the two entropies. 

Most of the plots are produced in 4 copies: one for each target class and one for all of them combined.


### Calibration
The calibration method chosen for the classification step (namely `filter` and `gnn`) is [Platt scaling][3]. Namely, the reliability distribution (ie. proportion of _True_ edges vs *edge_score*) is fitted with a logistic function
$$f(x) = \frac{1}{1+e^{a(x+b)}}$$
 To calibrate an `edge_classifier` model once it has been trained and infered, one should run
```bash
acorn calib {model}_calibrate.yaml
```
where `{model}` is either `filer` or `gnn`.
This outputs a calibration and a reliability plot after the Platt scaling has been done along with a `rescaling.txt` file. This file contains the parameters $a$ and $b$ of the logistic function fit.
This `calib` step DOES NOT infer new graphs. To infer new graph with calibrated scores, one has to put the `calibration` feature of the `{model}_infer.yaml` to `True` and redo the inference with:
```bash
acorn infer {model}_infer.yaml
```
## UQ/Calibration-specific parameters in  `.yaml` files $\dagger$
### `train` stage
None
### `infer` stage
In the `infer` stage `.yaml` files you will find the following extra parameters:
- `calibration`: If set to `True`, uses the calibration parameters to infer calibrated graph edge scores.
### `eval` stage
In the `eval` stage `.yaml` files you will find the following extra parameters:
- `calibration`: If set to `True` the associated efficiency plots will be saved in an associated "calibrated" folder.
- `n_train`: Correspond to the size of the train dataset. Used to display the number of train event used on the plots.
- `nb_MCD_passes`: Set the number of stochastic inference in the UQ MCD procedure.

[1]: https://arxiv.org/abs/1506.02142
[2]: https://www.cs.ox.ac.uk/people/yarin.gal/website/thesis/thesis.pdf
[3]: https://en.wikipedia.org/wiki/Platt_scaling
