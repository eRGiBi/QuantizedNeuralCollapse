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

    def _compute_class_means_and_covariances(self):
        """Compute the mean feature vector for each class and the
        within-class and between-class covariance matrices.
        """
        if self.features is None: self._extract_penultimate_features()

        feature_dim = self.features.shape[1]
        self.class_means = torch.zeros(self.num_classes, feature_dim)

        # Class means
        for c in range(self.num_classes):
            class_features = self.features[self.labels == c]
            self.class_means[c] = class_features.mean(dim=0)

        # Within-class covariance matrix - Sw
        self.Sw = torch.zeros(feature_dim, feature_dim)
        for c in range(self.num_classes):
            diff = self.features[self.labels == c] - self.class_means[c]
            self.Sw += diff.T @ diff
        self.Sw /= self.features.shape[0]

        # Between-class covariance matrix - Sb
        global_mean = self.features.mean(dim=0)
        self.Sb = torch.zeros(feature_dim, feature_dim)
        for c in range(self.num_classes):
            n_c = (self.labels == c).sum()
            diff = (self.class_means[c] - global_mean).unsqueeze(1)
            self.Sb += n_c * (diff @ diff.T)

        self.Sb /= self.features.shape[0]

    def test_nc1_variability_collapse(self):
        """NC1: Intra-class features collapse to their class means.

        measure by the trace of the within-class covariance matrix (Tr(Sw)).
        """
        if self.Sw is None: self._compute_class_means_and_covariances()

        collapse_ratio = torch.trace(self.Sw) / (torch.trace(self.Sb) + 1e-6)

        print("\n--- NC1: Variability Collapse ---")
        print(f"Trace of within-class covariance (Tr(Sw)): {torch.trace(self.Sw):.4f}")
        print(f"Trace of between-class covariance (Tr(Sb)): {torch.trace(self.Sb):.4f}")
        print(f"Collapse Ratio (Tr(Sw) / Tr(Sb)): {collapse_ratio:.4f}")

        return torch.trace(self.Sw).item(), collapse_ratio.item()

    def test_nc2_etf_class_means(self):
        """NC2: The class means converge to a simplex Equiangular Tight Frame (ETF).

        ||μ_c|| is constant and the angle between any two μ_i, μ_j is the same.
        """
        if self.class_means is None: self._compute_class_means_and_covariances()

        # Mean centering
        means_centered = self.class_means - self.class_means.mean(dim=0, keepdim=True)

        means_normalized = F.normalize(means_centered, p=2, dim=1)

        # Compute the Gram matrix of class means : M_μ = μ^T * μ
        gram_matrix = means_normalized @ means_normalized.T

        # For a perfect simplex ETF, off-diagonal elements should be -1/(K-1)
        K = self.num_classes

        # Create the ideal ETF Gram matrix
        ideal_off_diagonal = -1.0 / (K - 1)
        ideal_gram_matrix = torch.full((K, K), ideal_off_diagonal)
        ideal_gram_matrix.fill_diagonal_(1.0)

        # Calculate the deviation
        deviation = torch.norm(gram_matrix - ideal_gram_matrix) / torch.norm(ideal_gram_matrix)

        print("\n--- NC2: Class Means form a Simplex ETF ---")
        print(f"Deviation from ideal ETF structure for class means: {deviation:.4f}")

        return deviation.item()

    def test_nc3_self_dual_alignment(self):
        """NC3: The last layer classifier weights (W) align with the class means (μ).

        1. The classifier weights themselves also form a simplex ETF.
        2. The weights W are aligned with the class means μ.
        """
        if self.class_means is None: self._compute_class_means_and_covariances()

        W = self.model.classifier.weight.detach().cpu().T  # Shape: (feature_dim, num_classes)

        # Check if classifier weights form an ETF
        W_normalized = F.normalize(W, p=2, dim=0)
        gram_matrix_W = W_normalized.T @ W_normalized

        K = self.num_classes
        ideal_gram_matrix = torch.eye(K) - (1 - torch.eye(K)) / (K - 1)
        deviation_W = torch.norm(gram_matrix_W - ideal_gram_matrix) / torch.norm(ideal_gram_matrix)

        # Check alignment between W and μ
        means_normalized = F.normalize(self.class_means, p=2, dim=1).T  # Shape: (feature_dim, num_classes)

        # Cosine similarity between corresponding vectors
        # (W_norm.T @ mu_norm).diag() later
        cos_similarities = F.cosine_similarity(W_normalized, means_normalized, dim=0)

        avg_cos_similarity = torch.mean(cos_similarities)

        print("\n--- NC3: Self-Dual Alignment ---")
        print(f"Deviation from ideal ETF for classifier weights: {deviation_W:.4f}")
        print(f"Average cosine similarity between classifier weights and class means: {avg_cos_similarity:.4f}")

        return deviation_W.item(), avg_cos_similarity.item()

    def test_nc4_nearest_class_mean(self):
        """The classifier behaves like a nearest class-mean classifier."""
        if self.class_means is None: self._compute_class_means_and_covariances()

        model_preds = torch.argmax(
            self.features @ self.model.classifier.weight.T.cpu() + self.model.classifier.bias.cpu(), dim=1)

        # class-mean predictions using squared Euclidean distance = argmin
        dists = torch.cdist(self.features, self.class_means)
        ncm_preds = torch.argmin(dists, dim=1)

        agreement = (model_preds == ncm_preds).float().mean()

        ncm_accuracy = (ncm_preds == self.labels).float().mean()
        model_accuracy_on_features = (model_preds == self.labels).float().mean()

        print("\n--- NC4: Nearest Class-Mean Equivalence ---")
        print(f"Model accuracy (on extracted features): {model_accuracy_on_features:.4f}")
        print(f"Nearest Class-Mean classifier accuracy: {ncm_accuracy:.4f}")
        print(f"Agreement between model and NCM classifier: {agreement:.4f}")

        return agreement.item()


