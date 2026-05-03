import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Dict, Any
from pathlib import Path


def plot_metrics(self, save_path: str = "nc_metrics.png"):
    """Plot neural collapse metrics over training."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Neural Collapse Metrics During Training", fontsize=16)

    metrics_names = ["nc1", "nc2", "nc3", "nc4"]
    titles = [
        "NC1: Within-Class Variability",
        "NC2: Class Means Simplex ETF",
        "NC3: Self-Duality (Means-Weights Alignment)",
        "NC4: Classifier Weights Simplex ETF",
    ]

    for idx, (metric_name, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(
            metrics_history["epoch"],
            metrics_history[metric_name],
            marker="o",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value (lower is better)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Metrics plot saved to {save_path}")
    plt.close()



def save_training_image(log_data, metrics, output_path):
    """Generate and save training metrics visualization."""

    df = pd.DataFrame(log_data)

    skip_cols = ["epoch", "timestamp"]
    for col in df.columns:
        if col not in skip_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")


    # Filter to only metrics that exist in the dataframe
    metrics_to_plot = [m for m in metrics if m in df.columns]

    if not metrics_to_plot:
        print("No metrics found to plot.")
        return

    # Calculate grid size (aim for roughly square layout)
    n_metrics = len(metrics_to_plot)
    n_cols = 4  # 4 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Plot each metric
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_flat[idx]

        # Get valid (non-NaN) data for this metric
        valid_data = df[["epoch", metric]].dropna()

        if not valid_data.empty:
            sns.lineplot(
                data=valid_data,
                x="epoch",
                y=metric,
                marker="o",
                ax=ax,
                linewidth=2,
                markersize=6,
            )
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
        else:
            ax.text(
                0.5,
                0.5,
                f"{metric}\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_title(metric, fontsize=12, fontweight="bold")

        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(len(metrics_to_plot), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")

    # Print metric availability summary
    print("\nMetric Availability Summary:")
    print("-" * 50)
    for metric in metrics_to_plot:
        first_valid = df[metric].first_valid_index()
        if first_valid is not None:
            first_epoch = df.loc[first_valid, "epoch"]
            total_valid = df[metric].notna().sum()
            print(
                f"{metric:20s}: First value at epoch {first_epoch}, "
                f"{total_valid}/{len(df)} epochs have data"
            )
        else:
            print(f"{metric:20s}: No valid values")


if __name__ == "__main__":
    import pandas as pd

    # Define the path
    file_path = r"C:\Files\Egyetem\QuantizedNeuralCollapse\logs\20260107_013545_exp\metrics.csv"
    file_path = r"C:\Files\Egyetem\QuantizedNeuralCollapse\logs\20260107_191934_exp\metrics.csv"

    # Load the actual data from the CSV file
    log_data = pd.read_csv(file_path)


    metrics = [
        # "train_loss",
        # "train_accuracy",
        # "learning_rate",
        "nc1_pinv",
        # "nc1_svd",
        # "nc1_quot",
        # "nc1_cdnv",
        "nc2_etf_err",
        # "nc2g_dist",
        # "nc2g_log",
        "nc3_dual_err",
        # "nc3u_uni_dual",
        "nc4_agree",
        # "nc5_ood_dev",
    ]
    save_training_image(log_data, metrics=metrics, output_path = "viz/resnet.png")
