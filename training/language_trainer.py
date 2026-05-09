import torch
from torch import nn

from collapse_analysis.language_collapse_analyzer import LanguageNeuralCollapseAnalyzer
from experiment_logger import ExperimentLogger
from training.base_trainer import  BaseModelTrainer


class LanguageTrainer(BaseModelTrainer):
    """Trainer for classic language models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        ood_loader: torch.utils.data.DataLoader,
        config: dict,
        logger: ExperimentLogger
    ):
        self.max_seq_length = config.get("max_seq_length", None)

        self.nc_analyzer = LanguageNeuralCollapseAnalyzer(
           ood_loader, config
        )

        super().__init__(model, train_loader, validation_loader, ood_loader, config, logger)


    def _process_batch(self, batch_data):
        """Process a (x, y) tuple batch."""
        # Dataloader yields (input_ids, labels) tuples
        if isinstance(batch_data, (tuple, list)):
            x, y = batch_data
            attention_mask = None
            # Most of our in-repo char datasets already return next-token labels (y = x shifted by 1)
            labels_already_shifted = True
        else:
            # Dict-style fallback
            x = batch_data["input_ids"]
            y = batch_data.get("labels", x.clone())
            attention_mask = batch_data.get("attention_mask", None)
            # HF-style convention: labels are aligned with inputs, trainer performs the shift.
            labels_already_shifted = False

        x = x.to(self.device)
        y = y.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits = self.model(x)  # (B, T, vocab_size) — single forward pass

        # Pack extra info into a dict so BaseModelTrainer doesn't need changes.
        return logits, {
            "labels": y,
            "attention_mask": attention_mask,
            "labels_already_shifted": labels_already_shifted,
        }

    def _compute_loss_and_accuracy(self, logits, labels, criterion):
        """Next-token prediction loss and accuracy.

        Supports two label conventions:
        - tuple/list batches (x, y): y is already next-token labels (shifted by 1)
        - dict batches: labels are aligned with input_ids and must be shifted here
        """
        B, T, C = logits.shape

        attention_mask = None
        labels_already_shifted = None

        if isinstance(labels, dict):
            attention_mask = labels.get("attention_mask", None)
            labels_already_shifted = labels.get("labels_already_shifted", None)
            labels = labels.get("labels")

        if labels is None:
            return None, 0, 0

        if labels_already_shifted is None:
            # Default to shifting, unless overridden by config.
            labels_already_shifted = bool(self.config.get("labels_already_shifted", False))

        if labels_already_shifted:
            # logits[t] predicts labels[t] (next token), including the last position.
            shift_logits = logits.contiguous()          # (B, T, C)
            shift_labels = labels.contiguous()          # (B, T)
            shift_attn = attention_mask
        else:
            # HF-style: labels are aligned with inputs; predict the next token.
            shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, C)
            shift_labels = labels[..., 1:].contiguous()       # (B, T-1)
            shift_attn = attention_mask[..., 1:].contiguous() if attention_mask is not None else None

        # Optionally ignore the first N positions in each sequence.
        # Rationale: when training on random windows, early positions have short context
        # (missing tokens before the window start), making next-token labels ambiguous.
        # This can create an irreducible error floor (~97-99%) even when the model has
        # enough capacity to memorize. Masking early positions is a standard LM trick.
        ignore_first_n = int(self.config.get("loss_ignore_first_n", 0))

        # Flatten for loss computation
        shift_logits_flat = shift_logits.view(-1, C)
        shift_labels_flat = shift_labels.view(-1)

        # Build a flat position index aligned with shift_labels_flat
        seq_len_eff = shift_labels.shape[1]
        if ignore_first_n >= seq_len_eff:
            ignore_first_n = max(0, seq_len_eff - 1)
        pos = torch.arange(seq_len_eff, device=shift_labels.device).unsqueeze(0).expand(B, -1)
        pos_flat = pos.reshape(-1)

        valid_mask = shift_labels_flat != -100
        if shift_attn is not None:
            valid_mask = valid_mask & (shift_attn.view(-1) != 0)

        if ignore_first_n > 0:
            valid_mask = valid_mask & (pos_flat >= ignore_first_n)

        if valid_mask.sum() == 0:
            return None, 0, 0

        loss = criterion(shift_logits_flat[valid_mask], shift_labels_flat[valid_mask])

        # Accuracy on the SAME positions as the loss
        predicted = shift_logits_flat[valid_mask].argmax(dim=1)
        correct = predicted.eq(shift_labels_flat[valid_mask]).sum().item()
        total = valid_mask.sum().item()

        return loss, correct, total
    #
    #
    # def train(self, criterion, optimizer, scheduler):
    #
    #     model = self.model
    #     config = self.config
    #     train_loader = self.train_loader
    #     nc_analyzer = self.nc_analyzer
    #     validation_loader = self.validation_loader
    #     device = config["device"]
    #
    #     for epoch in range(config["epochs"]):
    #         model.train()
    #         total_loss = 0
    #         running_loss = 0
    #         correct = 0
    #         total = 0
    #
    #         for batch_idx, (x, y) in enumerate(train_loader):
    #             # Both x and y are already the right shape: (batch, seq_len)
    #             x = x.to(device)
    #             y = y.to(device)
    #
    #             # Forward pass
    #             logits = model(x)  # (batch, seq_len, vocab_size)
    #
    #             # Inside your train loop:
    #             logits = model(x)
    #             B, T, C = logits.shape
    #
    #             # Shift so we predict the NEXT token
    #             shift_logits = logits[:, :-1, :].contiguous()
    #             shift_labels = y[:, 1:].contiguous()
    #
    #             # Now flatten
    #             loss = criterion(shift_logits.view(-1, C), shift_labels.view(-1))
    #
    #             # # Reshape for loss computation
    #             #
    #             # logits = logits.view(B * T, C)
    #             # y = y.view(B * T)
    #
    #             # # Compute loss
    #             # loss = criterion(logits, y)
    #
    #             # Backward pass
    #             optimizer.zero_grad()
    #             loss.backward()
    #
    #             # Gradient clipping
    #             # if config.get("clip_grad", False):
    #             #     torch.nn.utils.clip_grad_norm_(
    #             #         model.parameters(),
    #             #         config.get("max_grad_norm", 1.0)
    #             #     )
    #
    #             optimizer.step()
    #
    #             # Track metrics
    #             running_loss += loss.item()
    #             pred = logits.argmax(dim=-1)
    #             correct += (pred == y).sum().item()
    #             total += y.numel()
    #
    #             if batch_idx % 100 == 0:
    #                 curr_loss = running_loss / (batch_idx + 1)
    #                 curr_acc = 100.0 * correct / total
    #                 print(f"Batch {batch_idx}: Loss={curr_loss:.4f}, Acc={curr_acc:.2f}%")
    #
    #         epoch_loss = total_loss / len(train_loader)
    #         epoch_acc = 100.0 * correct / total
    #
    #         print(f"\nEpoch {epoch + 1}/{config['epochs']}")
    #         print(f"  Loss: {epoch_loss:.4f}")
    #         print(f"  Accuracy: {epoch_acc:.2f}%")
    #
    #
    #         if (epoch + 1) % config["nc_freq"] == 0:
    #             print(f"\n  Running Neural Collapse Analysis...")
    #
    #             nc_metrics = nc_analyzer.analyze(
    #                 model=model,
    #                 train_loader=train_loader,
    #                 test_loader=validation_loader,
    #                 analysis_loader=self.ood_loader,
    #                 device=config["device"],
    #             )
    #
    #             print(f"\n  NC Metrics:")
    #             print(f"    NC1 (CDNV): {nc_metrics['nc1_cdnv']:.6f}")
    #             print(f"    NC1 (Collapse Ratio): {nc_metrics['nc1_quot']:.6f}")
    #             print(f"    NC2 (ETF Error): {nc_metrics['nc2_etf_err']:.6f}")
    #             print(f"    NC2g (Distance Var): {nc_metrics['nc2g_dist']:.6f}")
    #             print(f"    NC3 (Dual Error): {nc_metrics['nc3_dual_err']:.6f}")
    #             print(f"    NC4 (Agreement): {nc_metrics['nc4_agree']:.4f}")