# Core scripts and utilities

This directory contains the core scripts and utilities for the GNN4ITK framework. It is a simple backbone for running the stages (specified in [the stages directory](../stages/README.md)) using the command line. They are called from the command line using 
```bash
acorn [train, infer, eval] my-config-file.yaml
``` 
Example config files can be found in the [examples directory](../../examples/README.md).

List of changes made from original ACORN :
- Add of the `calib_stage.py` script to handle calibration of edge scoring models
- Modification of `entrypoin_stage*.py` scripts to handle `calib_stage` module
- Modification of `eval_stage.py` script to handle dropout parameters and UQ MCD procedure
- Modification of `infer_stage.py` script to handle calibration of edge score predictions with splines.
