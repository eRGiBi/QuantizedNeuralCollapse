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
        """Process a batch for language models."""
        # Expect batch to be dict with 'input_ids', 'attention_mask', 'labels'
        input_ids = batch_data["input_ids"].to(self.device)
        attention_mask = batch_data.get("attention_mask", None)
        labels = batch_data.get("labels", input_ids.clone()).to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if attention_mask is not None:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        return logits, labels

    def _compute_loss_and_accuracy(self, logits, labels, criterion):
        """Compute loss and accuracy for LLMs."""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)

        # Ignore padding tokens (typically -100)
        valid_mask = shift_labels_flat != -100

        if valid_mask.sum() == 0:
            return None, 0, 0

        loss = criterion(shift_logits_flat[valid_mask], shift_labels_flat[valid_mask])

        # Compute accuracy on valid tokens
        _, predicted = shift_logits_flat[valid_mask].max(1)
        correct = predicted.eq(shift_labels_flat[valid_mask]).sum().item()
        total = valid_mask.sum().item()

        return loss, correct, total


    def train(self, criterion, optimizer, scheduler):

        model = self.model
        config = self.config
        train_loader = self.train_loader
        nc_analyzer = self.nc_analyzer
        validation_loader = self.validation_loader
        device = config["device"]

        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0
            running_loss = 0
            correct = 0
            total = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                # Both x and y are already the right shape: (batch, seq_len)
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                logits = model(x)  # (batch, seq_len, vocab_size)

                # Inside your train loop:
                logits = model(x)
                B, T, C = logits.shape

                # Shift so we predict the NEXT token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()

                # Now flatten
                loss = criterion(shift_logits.view(-1, C), shift_labels.view(-1))

                # # Reshape for loss computation
                #
                # logits = logits.view(B * T, C)
                # y = y.view(B * T)

                # # Compute loss
                # loss = criterion(logits, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                # if config.get("clip_grad", False):
                #     torch.nn.utils.clip_grad_norm_(
                #         model.parameters(),
                #         config.get("max_grad_norm", 1.0)
                #     )

                optimizer.step()

                # Track metrics
                running_loss += loss.item()
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()

                if batch_idx % 100 == 0:
                    curr_loss = running_loss / (batch_idx + 1)
                    curr_acc = 100.0 * correct / total
                    print(f"Batch {batch_idx}: Loss={curr_loss:.4f}, Acc={curr_acc:.2f}%")

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total

            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            print(f"  Loss: {epoch_loss:.4f}")
            print(f"  Accuracy: {epoch_acc:.2f}%")


            if (epoch + 1) % config["nc_freq"] == 0:
                print(f"\n  Running Neural Collapse Analysis...")

                nc_metrics = nc_analyzer.analyze(
                    model=model,
                    train_loader=train_loader,
                    test_loader=validation_loader,
                    analysis_loader=self.ood_loader,
                    device=config["device"],
                )

                print(f"\n  NC Metrics:")
                print(f"    NC1 (CDNV): {nc_metrics['nc1_cdnv']:.6f}")
                print(f"    NC1 (Collapse Ratio): {nc_metrics['nc1_quot']:.6f}")
                print(f"    NC2 (ETF Error): {nc_metrics['nc2_etf_err']:.6f}")
                print(f"    NC2g (Distance Var): {nc_metrics['nc2g_dist']:.6f}")
                print(f"    NC3 (Dual Error): {nc_metrics['nc3_dual_err']:.6f}")
                print(f"    NC4 (Agreement): {nc_metrics['nc4_agree']:.4f}")