

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
            self.metrics_history["epoch"],
            self.metrics_history[metric_name],
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