from collapse_analysis.vision_collapse_analyzer import VisionNeuralCollapseAnalyzer
from training.base_trainer import BaseModelTrainer


class VisionTrainer(BaseModelTrainer):
    """Trainer for computer vision models."""

    # def _process_batch_cv(self, inputs, labels):
    #     outputs = self.model(inputs)
    #     return outputs, labels

    def __init__(self, model, train_loader, validation_loader, ood_loader, config, logger):

        self.nc_analyzer = VisionNeuralCollapseAnalyzer(
            ood_loader, config
        )


        super().__init__(model, train_loader, validation_loader, ood_loader, config, logger)

    def _process_batch(self, batch_data):
        """Process a batch for computer vision models."""
        inputs, labels = batch_data
        inputs = inputs.to(
            self.device,
            dtype=self.global_target_dtype
        )
        labels = labels.to(
            self.device,
            # dtype=self.global_target_dtype
        )

        outputs = self.model(inputs)
        return outputs, labels


    def _compute_loss_and_accuracy(self, outputs, labels, criterion):
        """Compute loss and accuracy for CV models."""
        loss = criterion(outputs.float(), labels.long())

        _, predicted = outputs.max(1)

        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)

        return loss, correct, total
