from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from experiment_logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Neural-collapse metric groups
# Keys must match exactly what BaseModelTrainer writes into epoch_log_data.
# ---------------------------------------------------------------------------

_NC_GROUPS: Dict[str, Dict[str, Any]] = {
    "NC1 – Within-Class Variability Collapse": {
        "metrics": {
            "nc1_pinv": "Σ_W / Σ_B  (pinv)",
            "nc1_svd": "Σ_W / Σ_B  (SVD)",
            "nc1_quot": "Variability quotient",
            "nc1_cdnv": "CDNV",
        },
        "lower_is_better": True,
        "filename": "nc1_variability.png",
        "ylabel": "NC1 value  (↓ collapse)",
    },
    "NC2 – Convergence to ETF": {
        "metrics": {
            "nc2_etf_err": "ETF error",
            "nc2g_dist": "Geodesic dist",
            "nc2g_log": "Log-map dist",
        },
        "lower_is_better": True,
        "filename": "nc2_etf.png",
        "ylabel": "NC2 value  (↓ ETF)",
    },
    "NC3 – Self-Duality (Weight–Feature Alignment)": {
        "metrics": {
            "nc3_dual_err": "Dual error",
            "nc3u_uni_dual": "Uniform dual",
        },
        "lower_is_better": True,
        "filename": "nc3_duality.png",
        "ylabel": "NC3 value  (↓ aligned)",
    },
    "NC4 & NC5 – Classifier Agreement & OOD Deviation": {
        "metrics": {
            "nc4_agree": "NCM agreement  (↑)",
            "nc5_ood_dev": "OOD deviation  (↓)",
        },
        "lower_is_better": None,  # mixed; annotated per-line
        "filename": "nc4_nc5.png",
        "ylabel": "Value",
    },
}


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class ModelAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        logger: "ExperimentLogger",
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = next(model.parameters()).device
        self.results: Dict = {}
        self._cache: Dict = {}

        # Mirror the experiment directory so all artefacts share a run folder.
        if logger.save_to_disk and logger.exp_dir is not None:
            self.output_dir: Optional[Path] = logger.exp_dir / "analysis"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ood_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict:
        """Run every analysis pass, persist results, and return the full dict."""
        self._print("=" * 60)
        self._print("Post-training analysis")
        self._print("=" * 60)

        self.results["architecture"] = self.analyze_architecture()
        self.results["weight_stats"] = self.analyze_weights()
        self.results["val_performance"] = self.evaluate(val_loader, criterion, tag="val")
        self.results["train_performance"] = self.evaluate(
            train_loader, criterion, tag="train"
        )
        self.results["calibration"] = self.calibration_analysis(val_loader)
        self.results["ood"] = self.ood_analysis(val_loader, ood_loader)
        self.results["gradient_flow"] = self.gradient_flow_analysis(
            val_loader, criterion
        )
        self.results["dead_neurons"] = self.dead_neuron_analysis(val_loader)

        self._persist()
        self.plot_all()
        return self.results

    # ------------------------------------------------------------------
    # 1. Architecture
    # ------------------------------------------------------------------

    def analyze_architecture(self) -> Dict:
        self._print("\n[1/7] Architecture")
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        layer_info = [
            {
                "name": name,
                "type": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters()),
            }
            for name, module in self.model.named_modules()
            if not list(module.children())  # leaf modules only
        ]

        result = {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "model_size_mb": total * 4 / 1024**2,  # assumes float32
            "num_layers": len(layer_info),
            "layers": layer_info,
        }
        self._print(f"  Total params    : {total:,}")
        self._print(f"  Trainable params: {trainable:,}")
        self._print(f"  Model size (MB) : {result['model_size_mb']:.2f}")
        return result

    # ------------------------------------------------------------------
    # 2. Weight statistics
    # ------------------------------------------------------------------

    def analyze_weights(self) -> Dict:
        self._print("\n[2/7] Weight statistics")
        stats = {}
        for name, param in self.model.named_parameters():
            d = param.detach().float()
            stats[name] = {
                "mean": d.mean().item(),
                "std": d.std().item(),
                "min": d.min().item(),
                "max": d.max().item(),
                "l2_norm": d.norm(2).item(),
                "sparsity": (d.abs() < 1e-6).float().mean().item(),
            }
        self._print(f"  Computed stats for {len(stats)} parameter tensors")
        return stats

    # ------------------------------------------------------------------
    # 3. Performance evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        tag: str = "val",
    ) -> Dict:
        self._print(f"\n[3/7] Performance ({tag})")
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        ignore_first_n = int(self.config.get("loss_ignore_first_n", 0))

        for batch in loader:
            inputs, labels = self._unpack_batch(batch)
            logits = self._forward(inputs)

            # ── Sequence model: logits [B, T, C], labels [B, T] ──────────
            if logits.dim() == 3 and labels.dim() == 2:
                B, T, C = logits.shape

                if labels.shape[1] == T:
                    shift_logits, shift_labels = logits, labels
                elif labels.shape[1] == T - 1:
                    shift_logits, shift_labels = logits[:, :-1, :], labels
                else:
                    shift_logits = logits[:, :-1, :]
                    shift_labels = labels[:, 1:]

                flat_logits = shift_logits.reshape(-1, C)
                flat_labels = shift_labels.reshape(-1)

                seq_len = shift_labels.shape[1]
                ign = min(ignore_first_n, max(0, seq_len - 1))
                pos_flat = (
                    torch.arange(seq_len, device=labels.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                    .reshape(-1)
                )
                valid = flat_labels != -100
                if ign > 0:
                    valid = valid & (pos_flat >= ign)
                if valid.sum() == 0:
                    continue

                loss = criterion(flat_logits[valid], flat_labels[valid])
                probs_seq = F.softmax(shift_logits, dim=-1)
                preds_flat = probs_seq.argmax(dim=-1).reshape(-1)[valid]
                labels_flat = flat_labels[valid]

                n_valid = int(valid.sum().item())
                total_loss += loss.item() * n_valid
                correct += preds_flat.eq(labels_flat).sum().item()
                total += n_valid
                all_preds.append(preds_flat.cpu())
                all_labels.append(labels_flat.cpu())
                all_probs.append(F.softmax(flat_logits[valid], dim=-1).cpu())

            # ── Classification: logits [B, C], labels [B] ────────────────
            else:
                loss = criterion(logits, labels)
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                total_loss += loss.item() * labels.size(0)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
        all_labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
        all_probs = torch.cat(all_probs) if all_probs else torch.empty(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else float("nan")
        confidence = (
            all_probs.max(dim=-1).values.mean().item() if all_probs.numel() else float("nan")
        )
        per_class = (
            {
                int(c): all_preds[all_labels == c]
                .eq(all_labels[all_labels == c])
                .float()
                .mean()
                .item()
                for c in all_labels.unique().tolist()
            }
            if all_labels.numel()
            else {}
        )

        result = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "mean_confidence": confidence,
            "num_samples": total,
            "per_class_accuracy": per_class,
        }
        self._print(f"  Accuracy  : {accuracy:.4f}")
        self._print(f"  Loss      : {avg_loss:.4f}")
        self._print(f"  Confidence: {confidence:.4f}")

        self._cache[(tag, "probs")] = all_probs
        self._cache[(tag, "labels")] = all_labels
        return result

    # ------------------------------------------------------------------
    # 4. Calibration (ECE)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibration_analysis(self, loader: DataLoader, n_bins: int = 15) -> Dict:
        self._print("\n[4/7] Calibration (ECE)")
        self.model.eval()

        ignore_first_n = int(self.config.get("loss_ignore_first_n", 0))
        all_conf, all_correct = [], []

        for batch in loader:
            inputs, labels = self._unpack_batch(batch)
            probs = F.softmax(self._forward(inputs), dim=-1)
            conf, preds = probs.max(dim=-1)

            if conf.dim() == 2 and labels.dim() == 2:
                B, T = conf.shape
                ign = min(ignore_first_n, max(0, T - 1))
                if ign > 0:
                    conf = conf[:, ign:]
                    preds = preds[:, ign:]
                    labels = labels[:, ign:]
                valid = labels != -100
                if valid.sum() == 0:
                    continue
                all_conf.append(conf[valid].cpu())
                all_correct.append(preds.eq(labels)[valid].cpu().float())
            else:
                all_conf.append(conf.cpu())
                all_correct.append(preds.eq(labels).cpu().float())

        confidences = torch.cat(all_conf) if all_conf else torch.empty(0)
        corrects = torch.cat(all_correct) if all_correct else torch.empty(0)
        bin_edges = torch.linspace(0, 1, n_bins + 1)

        ece = 0.0
        bin_data = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            avg_conf = confidences[mask].mean().item()
            avg_acc = corrects[mask].mean().item()
            ece += mask.float().mean().item() * abs(avg_conf - avg_acc)
            bin_data.append({"conf": avg_conf, "acc": avg_acc})

        self._cache[("calibration", "bins")] = bin_data
        self._print(f"  ECE: {ece:.4f}")
        return {"ece": ece, "bins": bin_data, "n_bins": n_bins}

    # ------------------------------------------------------------------
    # 5. OOD detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ood_analysis(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
    ) -> Dict:
        self._print("\n[5/7] OOD detection")
        self.model.eval()

        ignore_first_n = int(self.config.get("loss_ignore_first_n", 0))

        def _msp_scores(loader: DataLoader) -> torch.Tensor:
            scores = []
            for batch in loader:
                inputs, _ = self._unpack_batch(batch)
                msp = F.softmax(self._forward(inputs), dim=-1).max(dim=-1).values
                if msp.dim() == 2:
                    B, T = msp.shape
                    ign = min(ignore_first_n, max(0, T - 1))
                    msp = msp[:, ign:].reshape(-1)
                scores.append(msp.cpu())
            return torch.cat(scores) if scores else torch.empty(0)

        id_scores = _msp_scores(id_loader)
        ood_scores = _msp_scores(ood_loader)

        auroc = _compute_auroc(id_scores, ood_scores)
        fpr95 = _compute_fpr_at_tpr(id_scores, ood_scores, tpr=0.95)

        result = {
            "auroc": auroc,
            "fpr_at_95_tpr": fpr95,
            "id_mean_score": id_scores.mean().item() if id_scores.numel() else float("nan"),
            "ood_mean_score": ood_scores.mean().item() if ood_scores.numel() else float("nan"),
            "id_std_score": id_scores.std().item() if id_scores.numel() else float("nan"),
            "ood_std_score": ood_scores.std().item() if ood_scores.numel() else float("nan"),
        }
        self._print(f"  AUROC         : {auroc:.4f}")
        self._print(f"  FPR @ 95% TPR : {fpr95:.4f}")

        self._cache[("ood", "id_scores")] = id_scores
        self._cache[("ood", "ood_scores")] = ood_scores
        return result

    # ------------------------------------------------------------------
    # 6. Gradient flow
    # ------------------------------------------------------------------

    def gradient_flow_analysis(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        n_batches: int = 5,
    ) -> Dict:
        self._print("\n[6/7] Gradient flow")
        self.model.train()

        grad_norms: Dict[str, List[float]] = defaultdict(list)
        ignore_first_n = int(self.config.get("loss_ignore_first_n", 0))

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            inputs, labels = self._unpack_batch(batch)
            self.model.zero_grad()
            logits = self._forward(inputs)

            if logits.dim() == 3 and labels.dim() == 2:
                B, T, C = logits.shape
                flat_logits = logits.reshape(-1, C)
                flat_labels = labels.reshape(-1)
                seq_len = labels.shape[1]
                ign = min(ignore_first_n, max(0, seq_len - 1))
                pos_flat = (
                    torch.arange(seq_len, device=labels.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                    .reshape(-1)
                )
                valid = flat_labels != -100
                if ign > 0:
                    valid = valid & (pos_flat >= ign)
                if valid.sum() == 0:
                    continue
                loss = criterion(flat_logits[valid], flat_labels[valid])
            else:
                loss = criterion(logits, labels)

            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name].append(param.grad.norm(2).item())

        per_layer = {
            name: {
                "mean_grad_norm": float(np.mean(norms)),
                "std_grad_norm": float(np.std(norms)),
            }
            for name, norms in grad_norms.items()
        }

        vanishing = [k for k, v in per_layer.items() if v["mean_grad_norm"] < 1e-6]
        exploding = [k for k, v in per_layer.items() if v["mean_grad_norm"] > 100]
        self._print(f"  Vanishing gradient layers: {len(vanishing)}")
        self._print(f"  Exploding gradient layers: {len(exploding)}")

        self._cache[("gradient", "norms")] = per_layer
        self.model.eval()
        return {
            "per_layer": per_layer,
            "vanishing_layers": vanishing,
            "exploding_layers": exploding,
        }

    # ------------------------------------------------------------------
    # 7. Dead neuron analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dead_neuron_analysis(
        self,
        loader: DataLoader,
        n_batches: int = 10,
    ) -> Dict:
        self._print("\n[7/7] Dead neurons")
        self.model.eval()

        activation_sums: Dict[str, torch.Tensor] = {}
        hooks = []

        def _make_hook(name: str):
            def _hook(module, input, output):
                act = output.detach().float()
                if act.dim() > 2:
                    act = act.mean(dim=list(range(2, act.dim())))
                mean_act = act.mean(dim=0)
                if name in activation_sums:
                    activation_sums[name] += mean_act
                else:
                    activation_sums[name] = mean_act.clone()

            return _hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(_make_hook(name)))

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            self._forward(self._unpack_batch(batch)[0])

        for h in hooks:
            h.remove()

        per_layer = {
            name: {
                "dead_fraction": ((total / n_batches) < 1e-6).float().mean().item(),
                "num_units": total.numel(),
            }
            for name, total in activation_sums.items()
        }
        mean_dead = (
            float(np.mean([v["dead_fraction"] for v in per_layer.values()]))
            if per_layer
            else 0.0
        )
        self._print(f"  Mean dead fraction: {mean_dead:.4f}")
        return {"per_layer": per_layer, "mean_dead_fraction": mean_dead}

    # ------------------------------------------------------------------
    # Persist via ExperimentLogger
    # ------------------------------------------------------------------

    def _persist(self):
        """
        Route results back through ExperimentLogger:

        Flat scalars   → logger.log_metrics()   (appended to metrics.csv)
        Nested results → logger.save_results()  (results.json, weight_stats
                         omitted to keep the file human-readable)
        """
        val = self.results.get("val_performance", {})
        train = self.results.get("train_performance", {})
        ood = self.results.get("ood", {})
        cal = self.results.get("calibration", {})
        arch = self.results.get("architecture", {})
        gf = self.results.get("gradient_flow", {})
        dn = self.results.get("dead_neurons", {})

        flat_metrics = {
            "analysis_val_accuracy": val.get("accuracy"),
            "analysis_val_loss": val.get("loss"),
            "analysis_val_confidence": val.get("mean_confidence"),
            "analysis_train_accuracy": train.get("accuracy"),
            "analysis_ece": cal.get("ece"),
            "analysis_auroc": ood.get("auroc"),
            "analysis_fpr95": ood.get("fpr_at_95_tpr"),
            "analysis_id_mean_score": ood.get("id_mean_score"),
            "analysis_ood_mean_score": ood.get("ood_mean_score"),
            "analysis_total_params": arch.get("total_params"),
            "analysis_model_size_mb": arch.get("model_size_mb"),
            "analysis_vanishing_layers": len(gf.get("vanishing_layers", [])),
            "analysis_exploding_layers": len(gf.get("exploding_layers", [])),
            "analysis_mean_dead_fraction": dn.get("mean_dead_fraction"),
        }
        # _append_to_csv handles new column keys gracefully.
        self.logger.log_metrics(flat_metrics)

        self.logger.save_results(
            {k: v for k, v in self.results.items() if k != "weight_stats"}
        )

        if self.output_dir:
            self._print(f"\nAnalysis artefacts → {self.output_dir}")

    # ------------------------------------------------------------------
    # Training-history helpers (reads back from logger.log_data)
    # ------------------------------------------------------------------

    def _load_training_history(self) -> pd.DataFrame:
        """
        Return a DataFrame of epoch-level rows from logger.log_data.

        Rows that contain an "epoch" key are training rows; the final
        analysis row (no "epoch") is excluded automatically.  Any NC
        metric stored as the string "nan" is cast to float NaN so
        matplotlib skips it cleanly.
        """
        rows = [r for r in self.logger.log_data if "epoch" in r]
        if not rows:
            # Fallback: read from disk if available
            if self.logger.save_to_disk and self.logger.metrics_path.exists():
                df = pd.read_csv(self.logger.metrics_path)
                if "epoch" in df.columns:
                    rows = df[df["epoch"].notna()].to_dict("records")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Coerce "nan" strings → float NaN
        for col in df.columns:
            if col != "epoch":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("epoch").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Master plot dispatcher
    # ------------------------------------------------------------------

    def plot_all(self):
        """Generate and save every plot into output_dir/analysis/."""
        if self.output_dir is None:
            return  # dry-run mode

        history = self._load_training_history()

        # ── Model-level plots ────────────────────────────────────────────
        self._plot_training_curves(history)
        self._plot_validation_curves(history)
        self._plot_calibration()
        self._plot_ood_histogram()
        self._plot_gradient_flow()
        self._plot_confidence_histogram()
        self._plot_per_class_accuracy()
        self._plot_weight_norm_heatmap()

        # ── Neural-collapse plots (one per group) ────────────────────────
        for group_title, group_cfg in _NC_GROUPS.items():
            self._plot_nc_group(history, group_title, group_cfg)

        # ── NC summary dashboard ─────────────────────────────────────────
        self._plot_nc_dashboard(history)

        self._print(f"  Plots saved → {self.output_dir}")

    # ------------------------------------------------------------------
    # Training-curve plot
    # ------------------------------------------------------------------

    def _plot_training_curves(self, history: pd.DataFrame):
        if history.empty:
            return

        cols_present = {
            "train_loss": "Training loss",
            "train_accuracy": "Training accuracy (%)",
            "learning_rate": "Learning rate",
        }
        available = {k: v for k, v in cols_present.items() if k in history.columns}
        if not available:
            return

        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        epochs = history["epoch"].values
        for ax, (col, label) in zip(axes, available.items()):
            vals = history[col].values
            ax.plot(epochs, vals, marker="o", markersize=3, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Validation Curves", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.output_dir / "validation_curves.png", dpi=150)
        plt.close(fig)


    def _plot_validation_curves(self, history: pd.DataFrame):
        if history.empty:
            return

        cols_present = {
            "val_loss": "Validation loss",
            "val_accuracy": "Validation accuracy (%)",
            "learning_rate": "Learning rate",
        }
        available = {k: v for k, v in cols_present.items() if k in history.columns}
        if not available:
            return

        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        epochs = history["epoch"].values
        for ax, (col, label) in zip(axes, available.items()):
            vals = history[col].values
            ax.plot(epochs, vals, marker="o", markersize=3, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Validation Curves", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.output_dir / "validation_curves.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Neural-collapse group plot
    # ------------------------------------------------------------------

    def _plot_nc_group(
        self,
        history: pd.DataFrame,
        group_title: str,
        group_cfg: Dict,
    ):
        if history.empty:
            return

        metrics: Dict[str, str] = group_cfg["metrics"]
        available = {k: v for k, v in metrics.items() if k in history.columns}
        if not available:
            return

        # Find epochs where at least one NC metric was measured
        measured_mask = history[list(available.keys())].notna().any(axis=1)
        hist = history[measured_mask]
        if hist.empty:
            return

        epochs = hist["epoch"].values
        fig, ax = plt.subplots(figsize=(8, 4))

        colors = plt.cm.tab10.colors
        for idx, (col, label) in enumerate(available.items()):
            vals = hist[col].values
            valid = ~np.isnan(vals.astype(float))
            if not valid.any():
                continue
            ax.plot(
                epochs[valid],
                vals[valid],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=label,
                color=colors[idx % len(colors)],
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(group_cfg["ylabel"])
        ax.set_title(group_title, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / group_cfg["filename"], dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # NC summary dashboard  (2 × 2 grid of all NC groups)
    # ------------------------------------------------------------------

    def _plot_nc_dashboard(self, history: pd.DataFrame):
        if history.empty:
            return

        groups = list(_NC_GROUPS.items())
        n_rows, n_cols = 2, 2
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

        colors = plt.cm.tab10.colors

        for panel_idx, (group_title, group_cfg) in enumerate(groups):
            ax = fig.add_subplot(gs[panel_idx // n_cols, panel_idx % n_cols])
            metrics: Dict[str, str] = group_cfg["metrics"]
            available = {k: v for k, v in metrics.items() if k in history.columns}
            if not available:
                ax.set_visible(False)
                continue

            measured_mask = history[list(available.keys())].notna().any(axis=1)
            hist = history[measured_mask]
            if hist.empty:
                ax.set_visible(False)
                continue

            epochs = hist["epoch"].values
            for idx, (col, label) in enumerate(available.items()):
                vals = hist[col].values
                valid = ~np.isnan(vals.astype(float))
                if not valid.any():
                    continue
                ax.plot(
                    epochs[valid],
                    vals[valid],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=label,
                    color=colors[idx % len(colors)],
                )

            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel(group_cfg["ylabel"], fontsize=8)
            ax.set_title(group_title, fontsize=9, fontweight="bold")
            ax.legend(loc="best", fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        fig.suptitle("Neural Collapse Dashboard", fontsize=14, fontweight="bold", y=1.01)
        fig.savefig(
            self.output_dir / "nc_dashboard.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Calibration reliability diagram
    # ------------------------------------------------------------------

    def _plot_calibration(self):
        bins = self._cache.get(("calibration", "bins"))
        if not bins:
            return

        confs = [b["conf"] for b in bins]
        accs = [b["acc"] for b in bins]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.bar(confs, accs, width=0.05, alpha=0.7, label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ece = self.results["calibration"]["ece"]
        ax.set_title(f"Reliability Diagram  (ECE = {ece:.4f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "calibration.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # OOD score histogram
    # ------------------------------------------------------------------

    def _plot_ood_histogram(self):
        id_scores = self._cache.get(("ood", "id_scores"))
        ood_scores = self._cache.get(("ood", "ood_scores"))
        if id_scores is None or not id_scores.numel():
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: overlapping histograms
        axes[0].hist(
            id_scores.numpy(), bins=60, alpha=0.6, label="In-distribution", density=True
        )
        axes[0].hist(ood_scores.numpy(), bins=60, alpha=0.6, label="OOD", density=True)
        axes[0].set_xlabel("Max softmax score (MSP)")
        axes[0].set_ylabel("Density")
        auroc = self.results["ood"]["auroc"]
        fpr95 = self.results["ood"].get("fpr_at_95_tpr", float("nan"))
        axes[0].set_title(
            f"OOD Detection\nAUROC={auroc:.4f}  FPR@95={fpr95:.4f}"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: box plot comparison
        axes[1].boxplot(
            [id_scores.numpy(), ood_scores.numpy()],
            labels=["In-distribution", "OOD"],
            patch_artist=True,
            boxprops=dict(alpha=0.7),
        )
        axes[1].set_ylabel("Max softmax score (MSP)")
        axes[1].set_title("Score Distribution (Box)")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "ood_detection.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Gradient flow bar chart
    # ------------------------------------------------------------------

    def _plot_gradient_flow(self):
        grad_data = self._cache.get(("gradient", "norms"))
        if not grad_data:
            return

        names = list(grad_data.keys())
        means = np.array([grad_data[n]["mean_grad_norm"] for n in names])
        stds = np.array([grad_data[n]["std_grad_norm"] for n in names])
        x = np.arange(len(names))

        vanishing_thresh = 1e-6
        exploding_thresh = 100
        colors_bar = [
            "#d62728" if m > exploding_thresh else "#1f77b4" if m > vanishing_thresh else "#7f7f7f"
            for m in means
        ]

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.35), 4))
        ax.bar(x, means, yerr=stds, color=colors_bar, alpha=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.set_ylabel("Mean gradient L2 norm")
        ax.set_title(
            "Gradient Flow per Layer\n"
            "(grey = vanishing < 1e-6,  red = exploding > 100)"
        )
        ax.axhline(vanishing_thresh, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(exploding_thresh, color="red", linestyle="--", linewidth=0.8)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "gradient_flow.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Confidence histogram
    # ------------------------------------------------------------------

    def _plot_confidence_histogram(self):
        val_probs = self._cache.get(("val", "probs"))
        if val_probs is None or not val_probs.numel():
            return

        confs = val_probs.max(dim=-1).values.numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(confs, bins=60, edgecolor="white")
        ax.axvline(float(np.mean(confs)), color="red", linestyle="--", label=f"Mean={np.mean(confs):.3f}")
        ax.set_xlabel("Max softmax confidence")
        ax.set_ylabel("Count")
        ax.set_title("Validation Confidence Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "confidence_hist.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Per-class accuracy bar chart
    # ------------------------------------------------------------------

    def _plot_per_class_accuracy(self):
        per_class = self.results.get("val_performance", {}).get("per_class_accuracy", {})
        if not per_class:
            return

        classes = sorted(per_class.keys())
        accs = [per_class[c] * 100 for c in classes]
        overall = self.results["val_performance"].get("accuracy", 0) * 100

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), 4))
        bars = ax.bar(classes, accs, alpha=0.8)
        ax.axhline(overall, color="red", linestyle="--", linewidth=1.2, label=f"Overall {overall:.1f}%")

        # Colour bars below overall in a warning shade
        for bar, acc in zip(bars, accs):
            if acc < overall - 5:
                bar.set_color("#d62728")

        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Per-Class Validation Accuracy")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(self.output_dir / "per_class_accuracy.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Weight L2-norm heat-strip (one cell per parameter tensor)
    # ------------------------------------------------------------------

    def _plot_weight_norm_heatmap(self):
        weight_stats = self.results.get("weight_stats", {})
        if not weight_stats:
            return

        names = list(weight_stats.keys())
        l2_norms = np.array([weight_stats[n]["l2_norm"] for n in names])
        sparsity = np.array([weight_stats[n]["sparsity"] for n in names])

        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(names) * 0.35), 4), sharex=True)

        axes[0].bar(range(len(names)), l2_norms, alpha=0.8)
        axes[0].set_ylabel("L2 norm")
        axes[0].set_title("Weight L2 Norms per Parameter Tensor")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(range(len(names)), sparsity * 100, alpha=0.8, color="darkorange")
        axes[1].set_ylabel("Sparsity (%)")
        axes[1].set_title("Weight Sparsity (|w| < 1e-6)")
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=90, fontsize=5)
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "weight_norms.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Batch-unpack / forward helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        """Handle plain (inputs, labels) tuples and HuggingFace-style dicts."""
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[1]
            inputs = (
                {k: v.to(self.device) for k, v in inputs.items()}
                if isinstance(inputs, dict)
                else inputs.to(self.device)
            )
            return inputs, labels.to(self.device)
        # dict batch (HuggingFace / LanguageTrainer style)
        labels = batch["labels"].to(self.device)
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
        return inputs, labels

    def _forward(self, inputs) -> torch.Tensor:
        out = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
        return out if isinstance(out, torch.Tensor) else out[0]

    def _print(self, msg: str):
        print(msg)


# ---------------------------------------------------------------------------
# OOD metric helpers
# ---------------------------------------------------------------------------


def _compute_auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """AUROC treating in-distribution as the positive class."""
    if id_scores.numel() == 0 or ood_scores.numel() == 0:
        return float("nan")
    labels = torch.cat([torch.ones(len(id_scores)), torch.zeros(len(ood_scores))])
    scores = torch.cat([id_scores, ood_scores])
    labels_sorted = labels[scores.argsort(descending=True)].numpy()
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    if tps[-1] == 0 or fps[-1] == 0:
        return float("nan")
    return float(np.trapezoid(tps / tps[-1], fps / fps[-1]))


def _compute_fpr_at_tpr(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    tpr: float = 0.95,
) -> float:
    if id_scores.numel() == 0 or ood_scores.numel() == 0:
        return float("nan")
    threshold = torch.quantile(id_scores, 1 - tpr).item()
    return (ood_scores >= threshold).float().mean().item()
