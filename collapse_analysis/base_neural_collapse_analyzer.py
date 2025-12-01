import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_logger import ExperimentLogger


class BaseNeuralCollapseAnalyzer:
    """Analyze a neural network for Neural Collapse."""

    def __init__(
            self,
            # model,
            data_loader,
            num_classes,
            # logger: ExperimentLogger,
            device='cuda'
    ):
        # self.model = model
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.device = device
        # self.model.to(self.device)
        # self.model.eval()

        self.features = None
        self.labels = None
        self.class_means = None
        self.Sw = None
        self.Sb = None


    def analyze(
            self,
            model,
            train_loader,
            test_loader,
            ood_loader,
            device
    ):
        """"""
        pass

    def get_last_layer(self, model):
        last_linear_layer = None

        if hasattr(model, 'fc') and isinstance(
                model.fc, torch.nn.Linear
        ):
            return model.fc, model.fc.weight

        elif hasattr(model, 'classifier'):
            return model.classifier, model.classifier.weight

        else:

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear_layer = module

            if last_linear_layer is not None:
                if hasattr(last_linear_layer, 'weight') and last_linear_layer.weight is not None:
                    return last_linear_layer, last_linear_layer.weight

                else:
                    raise ValueError(f"Last identified nn.Linear layer ({name}) has no weights.")
            else:
                raise

            weights = model[-1].weight

    def _extract_penultimate_features(self, model):
        """Extract features from the penultimate layer of the network for the given dataset."""
        penultimate_features_list = []
        labels_list = []
        
        try:
            penultimate_layer = model.penultimate
        except AttributeError:
            print("Error: The model does not have a layer named 'penultimate'.")
            return

        def hook_fn(module, input, output):
            penultimate_features_list.append(output.detach().cpu())

        hook = penultimate_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            for data, targets in self.data_loader:
                data = data.to(self.device)
                model(data)
                labels_list.append(targets.cpu())

        hook.remove()

        self.features = torch.cat(penultimate_features_list, dim=0)
        self.labels = torch.cat(labels_list, dim=0) 
        
        print(f"Extracted {self.features.shape[0]} features of dimension {self.features.shape[1]}")

