import torch

from training.accumulators import (
    CovarAccumulator,
    DecAccumulator,
    MeanAccumulator,
    VarNormAccumulator
)
from collapse_analysis.nc_paper_metrics import *
from collapse_analysis.kernels import kernel_stats, log_kernel
from collapse_analysis.base_neural_collapse_analyzer import BaseNeuralCollapseAnalyzer


# class Features:
#     pass
#
# def hook(self, input, output):
#     Features.value = input[0].clone()


# dictionary to store the intermediate activation
activations = {}

def hook(module, input, output):
    activations['features'] = input[0].clone().detach()

class VisionNeuralCollapseAnalyzer(BaseNeuralCollapseAnalyzer):
    """"""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(args, kwargs)

    def analyze(
            self,
            model,
            train_loader,
            test_loader,
            ood_loader,
            device
    ):
        """Compute """
        print("--- Running Neural Collapse Analysis ---")

        metrics = {}

        classifier, weights = self.get_last_layer(model)
        classifier.register_forward_hook(hook)

        D_VECTORS = weights.shape[1]
        N_CLASSES = weights.shape[0]

        print(f"Detected Feature Dimension D_VECTORS: {D_VECTORS}")
        print(f"Detected Number of Classes N_CLASSES: {N_CLASSES}")

        with torch.no_grad():
            model.eval()

            # _, weights = self.get_last_layer(model)

            # NC collections
            mean_accum = MeanAccumulator(N_CLASSES, D_VECTORS, device)
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                # mean_accum.accumulate(Features.value, labels)
                mean_accum.accumulate(activations['features'], labels)
            means, mG = mean_accum.compute()

            var_norms_accum = VarNormAccumulator(N_CLASSES, D_VECTORS, "cuda", M=means)
            covar_accum = CovarAccumulator(N_CLASSES, D_VECTORS, "cuda", M=means)
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                var_norms_accum.accumulate(activations['features'], labels, means)
                # var_norms_accum.accumulate(Features.value, labels, means)
                covar_accum.accumulate(activations['features'], labels, means)
                # covar_accum.accumulate(Features.value, labels, means)

            var_norms, _ = var_norms_accum.compute()
            covar_within = covar_accum.compute()

            dec_accum = DecAccumulator(N_CLASSES, D_VECTORS, "cuda", M=means, W=weights)
            dec_accum.create_index(means)  # optionally use FAISS index for NCC
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                # mean embeddings (only) necessary again if not using FAISS index
                if dec_accum.index is None:
                    # dec_accum.accumulate(Features.value, labels, weights, means)
                    dec_accum.accumulate(activations['features'], labels, weights, means)
                else:
                    dec_accum.accumulate(activations['features'], labels, weights)
                    # dec_accum.accumulate(Features.value, labels, weights)

            ood_mean_accum = MeanAccumulator(N_CLASSES, D_VECTORS, "cuda")
            for i, (images, labels) in enumerate(ood_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # ood_mean_accum.accumulate(Features.value, labels)
                ood_mean_accum.accumulate(activations['features'], labels)

            _, mG_ood = ood_mean_accum.compute()

        metrics = {
            "nc1_pinv": covariance_ratio(covar_within, means, mG),
            "nc1_svd": covariance_ratio(covar_within, means, mG, "svd"),
            "nc1_quot": covariance_ratio(covar_within, means, mG, "quotient"),
            "nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
            "nc2_etf_err": simplex_etf_error(means, mG),
            "nc2g_dist": kernel_stats(means, mG, tile_size=64)[1],
            "nc2g_log": kernel_stats(means, mG, kernel=log_kernel, tile_size=64)[1],
            "nc3_dual_err": self_duality_error(weights, means, mG),
            "nc3u_uni_dual": similarities(weights, means, mG).var().item(),
            "nc4_agree": clf_ncc_agreement(dec_accum),
            "nc5_ood_dev": orthogonality_deviation(means, mG_ood),
        }

        # _, metrics['NC1_Collapse_Ratio'] = self.test_nc1_variability_collapse()
        #
        # metrics['NC2_ETF_Means_Deviation'] = self.test_nc2_etf_class_means()
        #
        # dev_W, cos_sim = self.test_nc3_self_dual_alignment()
        #
        # metrics['NC3_ETF_Weights_Deviation'] = dev_W
        # metrics['NC3_Alignment_Cosine_Sim'] = cos_sim
        #
        # metrics['NC4_NCM_Agreement'] = self.test_nc4_nearest_class_mean()

        print(f"Analysis Summary: {metrics}")

        return metrics

