# Example 3

## Setup

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files. These are TrackML events pulled from CodaLab, as well as a detector description file. Define your data directory MY_DATA_DIR with
```bash
data_dir=MY_DATA_DIR
```
You can choose whether to download a small dataset of 100 events `trackml_100_events.zip` (appropriate for toy model training, around 700Mb) or a larger dataset of 1500 events `trackml_1500_events.zip` (appropriate for full event training, around 11Gb):
```bash
wget https://portal.nersc.gov/cfs/m3443/dtmurnane/TrackML_Example/trackml_XXX_events.zip -P $data_dir/Example_3
wget https://portal.nersc.gov/cfs/m3443/dtmurnane/TrackML_Example/detectors.csv -P $data_dir/Example_3
unzip $data_dir/Example_3/trackml_XXX_events.zip -d $data_dir/Example_3
```

The location of this data, as well as all parameters controlling the GNN4ITk reconstruction chain, is specified in `yaml` config files. The data directory currently has a placeholder MY_DATA_DIR. Replace this with the actual data directory with
```bash
sed -i "s/MY_DATA_DIR/$data_dir/g" *.yaml
```

## Running the Example

The following commands will run the Example 1 pipeline. In general, they follow the pattern
```
train --> infer --> eval
``` 
where `train` is used to train a model, `infer` is used to apply the model to data, and `eval` is used to evaluate the performance of the model. If a model has already been trained (in the case of the Module Map in Example 1), we do not need to train it, only provide the model to the `infer` step.

**1.** First, we build our input data from the raw Athena events:
```bash
acorn infer data_reader.yaml
```

**2.** We start the graph construction by training the Metric Learning stage:
```bash
acorn train metric_learning_train.yaml
``` 

**3.** Then, we build graphs using the Metric Learning in inference:
```bash
acorn infer metric_learning_infer.yaml
```

**3a.** (Optional) We can examine the performance of the Metric Learning, by printing some efficiency plots:
```bash
acorn eval metric_learning_eval.yaml
```

**Optional**
----
If the graphs constructed from the metric learning stage are too large (they might be if, for instance, you are training on full events and the metric learning hasn't quite converged, or if you have a GPU with less than 10Gb memory), one can then apply a `Filter` to reduce the number of edges in the graph. This is done by training a `Filter` model.

**4.** 
```bash
acorn train filter_train.yaml
```

**5.** Then, we apply the `Filter` in inference:
```bash
acorn infer filter_infer.yaml
```
---

**6.** Then, we train the GNN (NOTE: if you trained the filter, then you should point the input directory to the filter stage directory, and comment out the "input_cut" configuration in the GNN training config file):
```bash
acorn train gnn_train.yaml
```

**7.** Once the GNN is trained (should take around half an hour), we apply the GNN in inference to produce a dataset of scored graphs:
```bash
acorn infer gnn_infer.yaml
```

**8.** Finally, we produce track candidates from the scored graphs:
```bash
acorn infer track_building_infer.yaml
```

**9.** And plot the performance
```bash
acorn eval track_building_eval.yaml
```
