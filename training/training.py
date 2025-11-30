import time
import tqdm

from experiment_logger import ExperimentLogger
from collapse_analysis.vision_collapse_analyzer import VisionNeuralCollapseAnalyzer
from collapse_analysis.language_collapse_analyzer import LanguageNeuralCollapseAnalyzer


# DEPRECATED


class ModelTrainer:
    """A class to train a PyTorch model."""
    def __init__(self, model, train_loader, analysis_loader, config: dict, logger: ExperimentLogger):
        self.model = model
        self.train_loader = train_loader
        self.analysis_loader = analysis_loader
        self.device = config["device"]
        self.logger = logger
        
        self.config = config
        
        # Detect model type
        self.model_type = config.get("task", "cv")

        # Gradient accumulation for limited memory
        self.grad_accumulation_steps = config.get("grad_accumulation_steps", 1)

        # For LLMs: max sequence length for truncation
        self.max_seq_length = config.get("max_seq_length", None)

        if config["task"] == "cv":
            self.nc_analyzer = VisionNeuralCollapseAnalyzer(
                model, analysis_loader, self.config["num_classes"], self.device
            )
        elif config["task"] == "nlp":
            self.nc_analyzer = LanguageNeuralCollapseAnalyzer(
                model, analysis_loader, self.config["num_classes"], self.device
            )


    def _process_batch_cv(self, inputs, labels):
        """Process a batch for computer vision models."""
        outputs = self.model(inputs)
        return outputs, labels

    def _process_batch_llm(self, batch):
        """Process a batch for language models."""
        # Expect batch to be dict with 'input_ids', 'attention_mask', 'labels'
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", input_ids.clone()).to(self.device)

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

    def _compute_loss_and_accuracy_llm(self, logits, labels, criterion):
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

    def _compute_loss_and_accuracy_cv(self, outputs, labels, criterion):
        """Compute loss and accuracy for CV models."""
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)

        return loss, correct, total

    def train(self, criterion, optimizer, scheduler):
        """Train the model for a number of epochs.

        Returns:
            nn.Module: The trained model.
        """
        model = self.model
        train_loader = self.train_loader
        device = self.device
        
        epochs = self.config["epochs"]
        analysis_freq = self.config['analysis_freq']

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
            for i, (inputs, labels) in enumerate(progress_bar):
                
                # Process batch based on model type
                if self.model_type == "cv":
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs, labels = self._process_batch_cv(inputs, labels)
                    
                    loss, batch_correct, batch_total = (
                        self._compute_loss_and_accuracy_cv(outputs, labels, criterion)
                )
                else:  # nlp
                    outputs, labels = self._process_batch_llm(batch_data)
                    
                    loss, batch_correct, batch_total = (
                        self._compute_loss_and_accuracy_llm(outputs, labels, criterion)
                    )
                    
                if loss is None:  # Skip if no valid tokens
                    continue
                
                # Gradient accumulation
                loss = loss / self.grad_accumulation_steps
                loss.backward()

                if (i + 1) % self.grad_accumulation_steps == 0:
                    # Gradient clipping for LLMs
                    if self.config["clip_grad"]:
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
                
                'NC1_Collapse_Ratio': "nan",
                'NC2_ETF_Means_Deviation': "nan",
                'NC3_ETF_Weights_Deviation': "nan",
                'NC3_Alignment_Cosine_Sim': "nan",
                'NC4_NCM_Agreement': "nan",
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
        training_time = (end_time - start_time)
        print(f"\nFinished Training. Total time: {training_time:.2f} sec.")
        
        if self.config.get("savemodel", False):
            self.logger.save_model(model)
                
        return model
    
    