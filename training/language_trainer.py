from experiment_logger import ExperimentLogger
from training.base_trainer import BaseModelTrainer


class LanguageTrainer(BaseModelTrainer):
    """Trainer for language models."""

    def __init__(
        self, model, trainloader, analysis_loader, config: dict, logger: ExperimentLogger
    ):
        super().__init__(model, trainloader, analysis_loader, config, logger)
        self.max_seq_length = config.get("max_seq_length", None)

    def _process_batch(self, batch_data):
        """Process a batch for language models."""
        # Expect batch to be dict with 'input_ids', 'attention_mask', 'labels'
        input_ids = batch_data["input_ids"].to(self.device)
        attention_mask = batch_data.get("attention_mask", None)
        labels = batch_data.get("labels", input_ids.clone()).to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass
        if attention_mask is not None:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids)

        # Get logits
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
