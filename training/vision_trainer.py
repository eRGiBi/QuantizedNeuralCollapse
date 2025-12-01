from training.base_trainer import BaseModelTrainer


class VisionTrainer(BaseModelTrainer):
    """Trainer for computer vision models."""

    def _process_batch(self, batch_data):
        """Process a batch for computer vision models."""
        inputs, labels = batch_data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        outputs = self.model(inputs)
        return outputs, labels

    def _compute_loss_and_accuracy(self, outputs, labels, criterion):
        """Compute loss and accuracy for CV models."""
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)

        return loss, correct, total


def _process_batch_cv(self, inputs, labels):
    """Process a batch for computer vision models."""
    outputs = self.model(inputs)
    return outputs, labels


def _compute_loss_and_accuracy_cv(self, outputs, labels, criterion):
    """Compute loss and accuracy for CV models."""
    loss = criterion(outputs, labels)
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)

    return loss, correct, total
