import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_nerfstudio_psnr(log_dir, output_path):
    # Find the tfevents file
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No tfevents files found in {log_dir}")
        return

    # Sort to get the latest
    event_files.sort(key=os.path.getmtime)
    event_file = event_files[-1]

    print(f"Reading events from {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Nerfstudio usually logs PSNR under 'Train PSNR' or 'Evaluation PSNR' or similar
    # In recent versions it's 'Train PSNR'. Fallback to 'Train Loss' if not found.
    tags = ea.Tags()['scalars']
    
    psnr_tag = None
    for tag in tags:
        if 'psnr' in tag.lower():
            psnr_tag = tag
            break
    
    fallback_tag = None
    if not psnr_tag:
        for tag in tags:
            if 'loss' in tag.lower() and 'dict' not in tag.lower():
                fallback_tag = tag
                break
    
    target_tag = psnr_tag or fallback_tag
    if not target_tag:
        print(f"Could not find PSNR or Loss tag in {tags}")
        return

    print(f"Plotting tag: {target_tag}")
    events = ea.Scalars(target_tag)
    iters = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(10, 5))
    plt.plot(iters, values, label=target_tag)
    plt.title(f'Nerfstudio {target_tag} vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel(target_tag)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PSNR plot to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    plot_nerfstudio_psnr(args.log_dir, args.output)
