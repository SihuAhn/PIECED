#./job_launcher.py
import sys
import os
import re
import subprocess
import argparse
from datetime import datetime

from trainer.main_continual import str_to_dict

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# Run with pieced/ as the working directory so that `trainer/main_continual.py` resolves.
PIECED_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PIECED_ROOT)

DATA_ROOT = os.path.join(PIECED_ROOT, "data")

start_time = datetime.now()

# Only these combos exist under pieced/bash_files right now.
data_name = "ntu60_xview"   # "ntu60_xsub", "ntu60_xview"
training_model = "PIECED"      # "FT", "PIECED"

# Map data_name -> dataset folder under DATA_ROOT
_dataset, _split = data_name.split("_", 1)        # e.g. "ntu60", "xsub"
data_dir_override = os.path.join(DATA_ROOT, _dataset, _split)

parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str, default=f'bash_files/{data_name}/{training_model}.sh')
parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--base_experiment_dir", type=str, default=f"./exp/{data_name}")
parser.add_argument("--data_dir", type=str, default=data_dir_override,
                    help="Overrides --data_dir from the bash script. Defaults to ./data/<dataset>/<split>.")
parser.add_argument("--gpu", type=str, default="RTX4090_24G")
parser.add_argument("--num_gpus", type=int, default=0)
parser.add_argument("--hours", type=int, default=20)
parser.add_argument("--requeue", type=int, default=0)

args = parser.parse_args()

# Sanity check: the data path must exist before we waste time launching training.
if not os.path.isdir(args.data_dir):
    print(f"[FATAL] data_dir does not exist: {args.data_dir}")
    sys.exit(1)

# load file
if os.path.exists(args.script):
    with open(args.script) as f:
        command = [line.strip().strip("\\").strip() for line in f.readlines()]
else:
    print(f"{args.script} does not exist.")
    exit()
assert (
    "--checkpoint_dir" not in command
), "Please remove the --checkpoint_dir argument, it will be added automatically"

# Replace the --data_dir value baked into the bash script with the absolute path
# that points into HiSDi_new/data.
joined = " ".join(command)
joined = re.sub(r"--data_dir\s+\S+", f"--data_dir {args.data_dir}", joined)
command = joined.split()

# collect args
command_args = str_to_dict(" ".join(command).split(" ")[2:])

# create experiment directory
if args.experiment_dir is None:
    args.experiment_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.experiment_dir += f"-{command_args['--name']}"
full_experiment_dir = os.path.join(args.base_experiment_dir, args.experiment_dir)
os.makedirs(full_experiment_dir, exist_ok=True)
print(f"Experiment directory: {full_experiment_dir}")
print(f"Data directory     : {args.data_dir}")

# add experiment directory to the command
command.extend(["--checkpoint_dir", full_experiment_dir])
command = " ".join(command)

print(f"Launching: {command}")

# run command
p = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stdout)
p.wait()

end_time = datetime.now()
elapsed = end_time - start_time

print(f"[START] {start_time:%Y-%m-%d %H:%M:%S}\n[ END ] {end_time:%Y-%m-%d %H:%M:%S}\n(elapsed: {str(elapsed).split('.')[0]})")
