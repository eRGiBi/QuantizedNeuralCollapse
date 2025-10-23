import time
import tqdm
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from experiment_logger import ExperimentLogger
from neural_collapse_analyzer import NeuralCollapseAnalyzer


class BaseModelTrainer(ABC):
    """Base class for model training."""

    def __init__(
        self, model: nn.Module, trainloader, analysis_loader, config: dict, logger: ExperimentLogger
    ):
        self.model = model
        self.trainloader = trainloader
        self.analysis_loader = analysis_loader
        self.device = config["device"]
        self.logger = logger
        self.config = config

        self.grad_accumulation_steps = config.get("grad_accumulation_steps", 1)

        self.nc_analyzer = NeuralCollapseAnalyzer(
            model, analysis_loader, self.config["num_classes"], self.device
        )

    @abstractmethod
    def _process_batch(self, batch_data):
        """Process a single batch of data."""
        pass

    @abstractmethod
    def _compute_loss_and_accuracy(self, outputs, labels, criterion):
        """Compute loss and accuracy for the batch."""
        pass

    def train(self, criterion, optimizer, scheduler):
        """Train the model for a number of epochs.

        Returns:
            nn.Module: The trained model.
        """
        model = self.model
        trainloader = self.trainloader
        device = self.device

        epochs = self.config["epochs"]
        analysis_freq = self.config["analysis_freq"]

        start_time = time.time()

        for epoch in range(epochs):
            model.train()

            running_loss, correct, total = 0.0, 0, 0

            progress_bar = tqdm.tqdm(
                trainloader,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=True,
                dynamic_ncols=True,
            )

            for i, batch_data in enumerate(progress_bar):

                outputs, labels = self._process_batch(batch_data)

                loss, batch_correct, batch_total = self._compute_loss_and_accuracy(
                    outputs, labels, criterion
                )

                if loss is None:  # Skip if no valid tokens
                    continue

                # Gradient accumulation
                loss = loss / self.grad_accumulation_steps

                loss.backward()

                if (i + 1) % self.grad_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.get("clip_grad", False):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.get("max_grad_norm", 1.0)
                        )

                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * self.grad_accumulation_steps
                correct += batch_correct
                total += batch_total

                # Update progress bar with current metrics
                if progress_bar.n > 0:
                    current_loss = running_loss / progress_bar.n
                    current_acc = 100.0 * correct / total if total > 0 else 0
                    progress_bar.set_postfix(
                        loss=f"{current_loss:.3f}", acc=f"{current_acc:.2f}%"
                    )

            # Calculate final epoch metrics
            epoch_loss = running_loss / len(trainloader)
            epoch_acc = 100.0 * correct / total

            epoch_log_data = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "learning_rate": (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                ),
                "NC1_Collapse_Ratio": "nan",
                "NC2_ETF_Means_Deviation": "nan",
                "NC3_ETF_Weights_Deviation": "nan",
                "NC3_Alignment_Cosine_Sim": "nan",
                "NC4_NCM_Agreement": "nan",
            }

            # Perform neural collapse analysis at specified frequency
            if (epoch + 1) % analysis_freq == 0 or (epoch + 1) == epochs:
                print(f"\nRunning neural collapse analysis...")
                try:
                    nc_metrics = self.nc_analyzer.analyze()
                    epoch_log_data.update(nc_metrics)
                except Exception as e:
                    print(f"Warning: NC analysis failed: {e}")

            if self.config["save"]:
                print(epoch_log_data)
                self.logger.log_training_metrics(epoch_log_data)

            if scheduler is not None:
                scheduler.step()

        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nFinished Training. Total time: {training_time:.2f} sec.")

        if self.config.get("savemodel", False):
            self.logger.save_model(model)

        return model