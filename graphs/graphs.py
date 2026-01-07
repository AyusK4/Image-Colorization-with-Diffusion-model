import argparse
import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_metrics() -> List[Dict]:
    """
    Load metrics preferentially from training_metrics.json.
    If not present, attempt to parse from 'training Metrics.txt' using convert.load_metrics_from_txt.
    """
    base = script_dir()
    json_path = os.path.join(base, "training_metrics.json")
    txt_path = os.path.join(base, "training Metrics.txt")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: parse from the text file if it exists
    if os.path.exists(txt_path):
        try:
            # Reuse the robust parser we added in convert.py
            from convert import load_metrics_from_txt  # type: ignore

            return load_metrics_from_txt(txt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse metrics from '{txt_path}': {e}")

    raise FileNotFoundError(
        "Could not find 'training_metrics.json' or 'training Metrics.txt' in the script folder."
    )


def to_dataframe(metrics: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        rows.append(
            {
                "epoch": m["epochs"][0],
                "train_loss": m["train_losses"][0],
                "val_loss": m["val_losses"][0],
                "learning_rate": m["learning_rates"][0],
                "epoch_time": m["epoch_times"][0],
                "best_val_loss": m["best_val_loss"],
                "best_epoch": m["best_epoch"],
            }
        )

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df


def plot_all(df: pd.DataFrame, outdir: str, show: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Combined 2x2 figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    ax1.plot(df["epoch"], df["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    ax1.plot(df["epoch"], df["val_loss"], "r-o", label="Val Loss", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Highlight best epoch if available
    # if "best_epoch" in df.columns and pd.notna(df["best_epoch"].iloc[-1]):
    #     be = int(df["best_epoch"].iloc[-1])
    #     ax1.axvline(be, color="gray", linestyle="--", alpha=0.6, label="Best Epoch")

    # Learning rate (log scale)
    ax2.semilogy(df["epoch"], df["learning_rate"], "g-o", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, which="both", alpha=0.3)

    # Epoch time
    ax3.plot(df["epoch"], df["epoch_time"], color="purple", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Epoch Training Time")
    ax3.grid(True, alpha=0.3)

    # Loss ratio (overfitting indicator)
    loss_ratio = df["train_loss"] / df["val_loss"].replace(0, pd.NA)
    ax4.plot(df["epoch"], loss_ratio, color="orange", linewidth=2)
    ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Train/Val Loss Ratio")
    ax4.set_title("Overfitting Indicator")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "all_plots.png"), dpi=300, bbox_inches="tight")

    # Individual figures
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], "b-o", label="Train")
    plt.plot(df["epoch"], df["val_loss"], "r-o", label="Val")
    # if "best_epoch" in df.columns and pd.notna(df["best_epoch"].iloc[-1]):
    #     be = int(df["best_epoch"].iloc[-1])
    #     plt.axvline(be, color="gray", linestyle="--", alpha=0.6, label="Best Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=300)

    plt.figure(figsize=(8, 5))
    plt.semilogy(df["epoch"], df["learning_rate"], "g-o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_rate.png"), dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["epoch_time"], color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("Epoch Training Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "epoch_time.png"), dpi=300)

    if show:
        plt.show()
    else:
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--outdir", default=os.path.join(script_dir(), "plots"), help="Output directory for images")
    parser.add_argument("--show", action="store_true", help="Show plots in a window")
    args = parser.parse_args()

    metrics = load_metrics()
    df = to_dataframe(metrics)
    plot_all(df, outdir=args.outdir, show=args.show)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()