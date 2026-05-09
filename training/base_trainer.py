import time
import tqdm
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from experiment_logger import ExperimentLogger
from quantization.q_controller import PRECISION_MAP


class BaseModelTrainer(ABC):
    """Base class for model training."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            ood_loader: torch.utils.data.DataLoader,
            config: dict,
            logger: ExperimentLogger
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.ood_loader = ood_loader
        self.device = config["device"]
        self.global_target_dtype = PRECISION_MAP.get(config["precision"], torch.float32)
        self.logger = logger
        self.config = config

        # Gradient accumulation for limited memory
        self.grad_accumulation_steps = config.get("grad_accumulation_steps", 1)

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
        # scaler = GradScaler()

        train_loader = self.train_loader
        val_loader = self.validation_loader
        ood_loader = self.ood_loader

        epochs = self.config["epochs"]
        analysis_freq = self.config["nc_freq"]
        val_freq = self.config.get("val_freq")

        # total_step = len(batch)
        # log_line = lambda epochs, i: f"Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}]"

        start_time = time.time()

        for epoch in range(epochs):
            self.model.to(self.device)
            self.model.train()

            running_loss, correct, total = 0.0, 0, 0

            progress_bar = tqdm.tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=True,
                dynamic_ncols=True,
            )

            optimizer.zero_grad(set_to_none=True)

            for i, batch_data in enumerate(progress_bar):

                # with autocast():
                outputs, labels = self._process_batch(batch_data)

                loss, batch_correct, batch_total = self._compute_loss_and_accuracy(
                    outputs, labels, criterion
                )

                if loss is None:  # Skip if no valid tokens
                    continue

                # Gradient accumulation
                loss_scaled = loss / max(1, self.grad_accumulation_steps)
                loss_scaled.backward()

                do_step = ((i + 1) % max(1, self.grad_accumulation_steps) == 0)
                if do_step:
                    # Gradient clipping
                    if self.config.get("clip_grad", False):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Track unscaled loss for logging
                running_loss += loss.item()
                correct += batch_correct
                total += batch_total

                # Update progress bar
                if progress_bar.n > 0:
                    current_loss = running_loss / progress_bar.n
                    current_acc = 100.0 * correct / total if total > 0 else 0
                    progress_bar.set_postfix(
                        loss=f"{current_loss:.3f}", acc=f"{current_acc:.2f}%"
                    )

            # Flush remainder gradients if epoch ended mid-accumulation
            if len(train_loader) % max(1, self.grad_accumulation_steps) != 0:
                if self.config.get("clip_grad", False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Epoch metrics
            epoch_loss = running_loss / max(1, len(train_loader))
            epoch_acc = 100.0 * correct / total

            val_loss = float('nan')
            val_acc = float('nan')

            if (epoch + 1) % val_freq == 0 or (epoch + 1) == epochs:
                self.model.eval()
                val_running_loss, val_correct, val_total = 0.0, 0, 0

                with torch.no_grad():
                    for val_batch_data in val_loader:
                        val_outputs, val_labels = self._process_batch(val_batch_data)
                        v_loss, v_batch_correct, v_batch_total = self._compute_loss_and_accuracy(
                            val_outputs, val_labels, criterion
                        )

                        if v_loss is not None:
                            val_running_loss += v_loss.item()
                            val_correct += v_batch_correct
                            val_total += v_batch_total

                if val_total > 0:
                    val_loss = val_running_loss / max(1, len(val_loader))
                    val_acc = 100.0 * val_correct / val_total
                    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            epoch_log_data = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                ),
                "nc1_pinv": "nan",
                "nc1_svd": "nan",
                "nc1_quot": "nan",
                "nc1_cdnv": "nan",
                "nc2_etf_err": "nan",
                "nc2g_dist": "nan",
                "nc2g_log": "nan",
                "nc3_dual_err": "nan",
                "nc3u_uni_dual": "nan",
                "nc4_agree": "nan",
                "nc5_ood_dev": "nan",
                # "NC1_Collapse_Ratio": "nan",
                # "NC2_ETF_Means_Deviation": "nan",
                # "NC3_ETF_Weights_Deviation": "nan",
                # "NC3_Alignment_Cosine_Sim": "nan",
                # "NC4_NCM_Agreement": "nan",
            }

            # Perform neural collapse analysis
            if (epoch + 1) % analysis_freq == 0 or (epoch + 1) == epochs or epoch == 0:
                print(f"\nRunning neural collapse analysis")

                nc_metrics = self.nc_analyzer.analyze(
                    self.model,
                    train_loader=train_loader,
                    validation_loader=self.validation_loader, #self.test_loader,
                    ood_loader=self.ood_loader,
                    device=self.device
                )
                epoch_log_data.update(nc_metrics)

            print(epoch_log_data)

            if self.config["save"]:
                self.logger.log_metrics(epoch_log_data)

            if scheduler is not None:
                scheduler.step()

        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nFinished Training. Total time: {training_time:.2f} sec.")

        if self.config.get("savemodel", False):
            self.logger.save_model(self.model)

        return self.model
