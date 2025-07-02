import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_train", type=bool)

args = parser.parse_args()
n_train = args.n_train

global_yaml_path = "/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/pt>1GeV/"
save_yaml_path = "/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/pt>1GeV/"

for pipeline_step in ["metric_learning", "filter", "gnn", "track_building"]:
    for model_step in ["_eval", "_calibrate"]:
        yaml_path = global_yaml_path + f"{pipeline_step}" + f"{model_step}.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                yaml_file = yaml.safe_load(file)
            yaml_file["n_train"] = int(n_train)

            # Extract the filename from the original path and use it with the save path
            filename = os.path.basename(yaml_path)
            save_file_path = os.path.join(save_yaml_path, filename)
            
            # Save to the new location
            with open(save_file_path, 'w') as file:
                yaml.safe_dump(yaml_file, file)