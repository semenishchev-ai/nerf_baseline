import os
import re
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs = []
    psnrs = []
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return epochs, psnrs
    with open(log_path, 'r') as f:
        lines = f.readlines()
    current_epoch = None
    for line in lines:
        m_eval = re.search(r'\+\+>\s*Evaluate at epoch (\d+)', line)
        if m_eval:
            current_epoch = int(m_eval.group(1))
        m_psnr = re.search(r'PSNR\s*=\s*([0-9\.]+)', line)
        if m_psnr and current_epoch is not None:
            psnrs.append(float(m_psnr.group(1)))
            epochs.append(current_epoch)
            current_epoch = None
    return epochs, psnrs

if __name__ == "__main__":
    logs_dir = "/home/a_semenishchev/VS/iter_2/modified_models/logs"
    experiments = {
        "Baseline (30k iters)": os.path.join(logs_dir, "baseline_30k", "train.log"),
        "New Model V1 (30k iters)": os.path.join(logs_dir, "new_model_v1_30k", "train.log")
    }
    plt.figure(figsize=(10, 6))
    for label, log_path in experiments.items():
        epochs_raw, psnrs_raw = parse_log(log_path)
        if epochs_raw:
            seen_epochs = set()
            epochs = []
            psnrs = []
            for e, p in zip(epochs_raw, psnrs_raw):
                if e not in seen_epochs:
                    seen_epochs.add(e)
                    epochs.append(e)
                    psnrs.append(p)
            data = sorted(zip(epochs, psnrs))
            epochs = [x[0] for x in data]
            psnrs = [x[1] for x in data]
            plt.plot(epochs, psnrs, marker='o', label=f"{label} (Final Val: {psnrs[-1]:.2f})")
    plt.title("Val PSNR over Epochs")
    plt.xlabel("Epochs (100 iters. / epoch)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(logs_dir, "comparative_psnr_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_path}")
