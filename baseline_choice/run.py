import argparse
import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Unified NeRF Baseline Runner")
    parser.add_argument("--model", type=str, required=True, choices=["torch_ngp", "hash_nerf", "nerfacto", "instant_ngp", "tensorf"], help="Model to run")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--data", type=str, default="./data/lego", help="Path to dataset")
    parser.add_argument("--iters", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--workspace", type=str, default=None, help="Workspace for outputs (default: logs/[model])")

    args = parser.parse_args()
    
    # Standardize workspace
    if args.workspace is None:
        args.workspace = os.path.join("logs", args.model)
    
    # Add baseline_choice to PYTHONPATH so sub-scripts can import 'common'
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

    # Ensure log directory exists
    os.makedirs(args.workspace, exist_ok=True)
    log_file_path = os.path.join(args.workspace, "train.log")
    print(f"Logging to {log_file_path}")

    with open(log_file_path, "w") as log_file:
        if args.model == "torch_ngp":
            cmd = ["python", "torch_ngp/main_nerf.py", args.data, "--workspace", args.workspace, "-O", "--iters", str(args.iters), "--bound", "1.0", "--scale", "0.8", "--dt_gamma", "0"]
            if args.mode == "eval":
                cmd.append("--test")
            subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

        elif args.model == "hash_nerf":
            # Pass smaller logging intervals for better visibility in short runs
            if args.mode == "train":
                cmd = ["python", "hash_nerf/run_nerf.py", "--config", "hash_nerf/configs/lego.txt", 
                       "--basedir", args.workspace, "--datadir", args.data, "--expname", "run", 
                       "--N_iters", str(args.iters), "--white_bkgd", "--lrate", "0.01", "--lrate_decay", "10", 
                       "--finest_res", "512", "--i_print", "10", "--i_img", "50", "--i_weights", "100"]
            else:
                cmd = ["python", "hash_nerf/run_nerf.py", "--config", "hash_nerf/configs/lego.txt", 
                       "--basedir", args.workspace, "--datadir", args.data, "--expname", "run", 
                       "--render_only", "--render_test", "--white_bkgd", "--lrate", "0.01", "--lrate_decay", "10", 
                       "--finest_res", "512"]
            subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

        elif args.model in ["nerfacto", "instant_ngp", "tensorf"]:
            model_name = args.model.replace("_", "-")
            if args.mode == "train":
                # Using --vis tensorboard and setting experiment-name for cleaner logs
                cmd = ["ns-train", model_name, 
                       "--max-num-iterations", str(args.iters), 
                       "--vis", "tensorboard", 
                       "--output-dir", "logs", 
                       "--experiment-name", args.model,
                       "--viewer.quit-on-train-completion", "True",
                       "blender-data", "--data", args.data]
            else:
                # Look for config.yml in the workspace
                # Nerfstudio structure: logs/[exp_name]/[model_name]/[timestamp]/config.yml
                import glob
                config_search_path = os.path.join("logs", args.model, "**", "config.yml")
                configs = glob.glob(config_search_path, recursive=True)
                if not configs:
                    print(f"Error: Config not found using pattern {config_search_path}. Run training first.")
                    sys.exit(1)
                configs.sort(key=os.path.getmtime)
                cmd = ["ns-eval", "--load-config", configs[-1], "--output-path", os.path.join("logs", args.model, "eval.json")]
            subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            
            # Post-training plot generation for Nerfstudio
            try:
                import glob
                # Find any folder under logs/[args.model] that contains tfevents
                event_files = glob.glob(os.path.join("logs", args.model, "**", "events.out.tfevents.*"), recursive=True)
                if event_files:
                    # Get the directory of the latest event file
                    event_files.sort(key=os.path.getmtime)
                    latest_event_file = event_files[-1]
                    event_dir = os.path.dirname(latest_event_file)
                    plot_cmd = ["python", "common/plot_nerfstudio.py", "--log_dir", event_dir, "--output", os.path.join(args.workspace, "psnr_plot.png")]
                    subprocess.run(plot_cmd, env=env)
            except Exception as e:
                print(f"Warning: Could not generate Nerfstudio plot: {e}")

if __name__ == "__main__":
    main()
