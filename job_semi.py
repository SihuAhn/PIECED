#./job_semi.py
"""
Semi-supervised linear evaluation launcher for PIECED.

Wraps `bash_files/{data_name}/semi_{training_model}.sh` and injects:
  - SEMI_RATIO       (label fraction, e.g. 0.01 / 0.1 / 1.0)
  - CHECKPOINT_DIR   (output dir for the run)
  - DATA_DIR         (local ./data/<dataset>/<split>)
  - CUDA_VISIBLE_DEVICES
"""
import sys
import os
import subprocess
import argparse
from datetime import datetime

# Run with pieced/ as cwd so `trainer/main_linear.py` resolves.
PIECED_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PIECED_ROOT)

DATA_ROOT = os.path.join(PIECED_ROOT, "data")

# Defaults — edit here or override on the CLI.
data_name = "ntu60_xsub"       # "ntu60_xsub", "ntu60_xview"
training_model = "PIECED"      # "FT", "PIECED"
DEFAULT_GPU = "5"
DEFAULT_RATIOS = [1.0]         # e.g. [0.01, 0.1, 1.0]


def run_semi_task(ratio, base_output_dir, gpu_id, script_path, data_dir):
    start_time = datetime.now()
    percent = int(ratio * 100)

    timestamp = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = os.path.join(base_output_dir, f"{percent}percent", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting {percent}% Evaluation")
    print(f"   - Ratio:    {ratio}")
    print(f"   - GPU:      {gpu_id}")
    print(f"   - Script:   {script_path}")
    print(f"   - Data:     {data_dir}")
    print(f"   - Log Dir:  {experiment_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["SEMI_RATIO"] = str(ratio)
    env["CHECKPOINT_DIR"] = experiment_dir
    env["DATA_DIR"] = data_dir

    cmd = f"bash {script_path}"
    try:
        subprocess.run(cmd, shell=True, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {percent}% evaluation: {e}")
        sys.exit(1)

    elapsed = datetime.now() - start_time
    print(f"Finished {percent}% Eval. (Elapsed: {str(elapsed).split('.')[0]})")


def main():
    _dataset, _split = data_name.split("_", 1)
    default_script = f"bash_files/{data_name}/semi_{training_model}.sh"
    default_data_dir = os.path.join(DATA_ROOT, _dataset, _split)

    parser = argparse.ArgumentParser(description="Run Semi-Supervised Linear Eval for PIECED")
    parser.add_argument("--script", type=str, default=default_script,
                        help=f"Shell script path (default: {default_script})")
    parser.add_argument("--out_dir", type=str, default=f"./exp/semi_eval/{data_name}",
                        help="Base directory for saving logs")
    parser.add_argument("--gpu", type=str, default=DEFAULT_GPU,
                        help=f"GPU ID to use (default: {DEFAULT_GPU})")
    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                        help="Dataset directory (default ./data/<dataset>/<split>)")
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                        help=f"Label ratios to evaluate (default: {DEFAULT_RATIOS})")
    args = parser.parse_args()

    if not os.path.exists(args.script):
        print(f"[FATAL] Shell script not found: {args.script}")
        sys.exit(1)
    if not os.path.isdir(args.data_dir):
        print(f"[FATAL] data_dir does not exist: {args.data_dir}")
        sys.exit(1)

    print(f"Using GPU:     {args.gpu}")
    print(f"Target Script: {args.script}")
    print(f"Ratios:        {args.ratios}")

    for ratio in args.ratios:
        run_semi_task(ratio, args.out_dir, args.gpu, args.script, args.data_dir)

    print("\nAll Semi-Supervised Evaluations Completed Successfully.")


if __name__ == "__main__":
    main()
