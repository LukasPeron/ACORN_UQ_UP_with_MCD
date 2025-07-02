import yaml
import os

def metric_learning_yaml():
    """
    This function modifies the yaml files for the metric learning step.
    It sets the number of epochs and other parameters.
    """
    pass


def filter_yaml():
    """
    This function filters the yaml files based on the specified conditions.
    It modifies the yaml files and saves them to a new location.
    """
    pass


def gnn_yaml():
    """
    This function modifies the yaml files for the GNN step.
    It sets the number of epochs and other parameters.
    """
    pass


def calibrate_yaml():
    """
    This function modifies the yaml files for the calibration step.
    It sets the number of epochs and other parameters.
    """
    pass


def track_building_yaml():
    """
    This function modifies the yaml files for the track building step.
    It sets the number of epochs and other parameters.
    """
    pass


for n_train in [50, 100, 200, 400, 800, 1400]:
    global_yaml_path = f"/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/{n_train}/"
    global_input_dir = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/"
    save_yaml_path = "/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/" + f"{n_train}/"

    for pipeline_step in ["metric_learning", "filter", "gnn", "track_building"]:
        for model_step in ["_train", "_infer", "_eval", "_calibrate"]:
            yaml_path = global_yaml_path + f"{pipeline_step}" + f"{model_step}.yaml"
            if os.path.exists(yaml_path):
                print(f"Processing {yaml_path}")
                with open(yaml_path, 'r') as file:
                    yaml_file = yaml.safe_load(file)
                yaml_file["devices"] = 4
                yaml_file["nodes"] = 1
                if pipeline_step =="filter":
                    yaml_file["input_dir"] = global_input_dir+"metric_learning/"
                    yaml_file["max_epochs"] = 250
                elif pipeline_step == "gnn":
                    yaml_file["input_dir"] = global_input_dir+"filter/"
                    yaml_file["max_epochs"] = 250
                elif pipeline_step == "track_building_":
                    yaml_file["input_dir"] = global_input_dir+"gnn/"
                
                yaml_file["stage_dir"] = global_input_dir + f"{pipeline_step}/"
                yaml_file["project"] = f"UQ_MCD_all_pt_{pipeline_step}_{n_train}"
                if model_step == "_eval" or model_step == "_calibrate":
                    yaml_file["data_split"] = [0, 50, 50]
                else:
                    yaml_file["data_split"] = [int(n_train), 50, 50]
                yaml_file["n_train"] = int(n_train)
                if pipeline_step == "filter" or pipeline_step == "gnn":
                    yaml_file["calibration"] = False
                    yaml_file["MCDropout"] = True
                    yaml_file["dropout"] = 0.1
                    yaml_file["nb_MCD_passes"] = 100

                # Extract the filename from the original path and use it with the save path
                filename = os.path.basename(yaml_path)
                save_file_path = os.path.join(save_yaml_path, filename)
                
                # Save to the new location
                with open(save_file_path, 'w') as file:
                    yaml.safe_dump(yaml_file, file)