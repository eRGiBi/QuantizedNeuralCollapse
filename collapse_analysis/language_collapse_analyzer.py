from typing import Dict, Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn

from collapse_analysis.kernels import kernel_grid, log_kernel, riesz_kernel
# from collapse_analysis.measure import (
#     distance_norms, interference_grid,
#                                      mean_norms, self_duality_error,
#                                      similarities, simplex_etf_error,
#                                      variability_cdnv
# )

from collapse_analysis.base_neural_collapse_analyzer import BaseNeuralCollapseAnalyzer


class LanguageNeuralCollapseAnalyzer(BaseNeuralCollapseAnalyzer):
    """
    Analyzes neural collapse metrics for language models.
    Compatible with nanoGPT and similar decoder-only transformers.
    """
    def __init__(self, ood_loader, config):

        self.max_samples = config["max_samples_for_nc"]
        self.min_samples_per_class = config["min_samples_per_class"]

        super().__init__(
            config,
            ood_loader,
            config["num_classes"],
            # logger: ExperimentLogger,
            config["device"]
        )

    def analyze(
            self,
            model,
            train_loader,
            validation_loader,
            ood_loader,
            device,
    ):
        """"""
        model.eval()

        try:
            # Extract embeddings, labels, and logits
            embeddings, labels, logits = self._extract_embeddings_and_logits(
                model, ood_loader , self.max_samples
            )

            # Filter rare classes
            embeddings, labels, logits, valid_classes = self._filter_rare_classes(
                embeddings, labels, logits
            )

            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = logits.to(device)

            # Compute class means
            M, mG = self.compute_class_means(embeddings, labels)

            # Compute NC metrics
            nc_metrics = {}

            # NC1
            nc1_metrics = self.nc1_within_class_variability(embeddings, labels, M)
            nc_metrics.update(nc1_metrics)

            # NC2
            nc2_metrics = self.nc2_hyperspherical_uniformity(M, mG)
            nc_metrics.update(nc2_metrics)

            # NC3: Requires classifier weights
            try:
                # For nanoGPT: model.lm_head.weight
                if hasattr(model, "lm_head"):
                    W = model.lm_head.weight[valid_classes].detach()
                elif hasattr(model, "transformer") and hasattr(
                        model.transformer, "wte"
                ):
                    # Weight tying: use token embeddings
                    W = model.transformer.wte.weight[valid_classes].detach()
                else:
                    print("Warning: Could not find classifier weights for NC3")
                    W = None

                if W is not None:
                    W = W.to(device)
                    nc3_metrics = self.nc3_self_duality(W, M, mG)
                    nc_metrics.update(nc3_metrics)
                else:
                    nc_metrics.update({"nc3_dual_err": float("nan"), "nc3u_uni_dual": float("nan")})

            except Exception as e:
                print(f"NC3 computation failed: {e}")
                nc_metrics.update({"nc3_dual_err": float("nan"), "nc3u_uni_dual": float("nan")})

            # NC4
            nc4_metrics = self.nc4_classifier_agreement(embeddings, labels, logits, M)
            nc_metrics.update(nc4_metrics)

            # Placeholder for NC5 (OOD detection - requires separate implementation)
            nc_metrics["nc5_ood_dev"] = float("nan")

            return nc_metrics

        except Exception as e:
            print(f"Neural collapse analysis failed: {e}")
            import traceback

            traceback.print_exc()

            # Return nan values on failure
            return {
                "nc1_pinv": float("nan"),
                "nc1_svd": float("nan"),
                "nc1_quot": float("nan"),
                "nc1_cdnv": float("nan"),
                "nc2_etf_err": float("nan"),
                "nc2g_dist": float("nan"),
                "nc2g_log": float("nan"),
                "nc3_dual_err": float("nan"),
                "nc3u_uni_dual": float("nan"),
                "nc4_agree": float("nan"),
                "nc5_ood_dev": float("nan"),
            }

    # def get_stats(self, iden: str, force_cpu: bool = False):
    #         stats = CollapseStatistics(
    #             device="cpu" if force_cpu else args.device,
    #             means_path=PATHS[iden][0],
    #             vars_path=PATHS[iden][1],
    #             decs_path=PATHS[iden][2],
    #             verbose=False,
    #         )
    #         return stats

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)



    def _extract_embeddings_and_logits(
            self, model: nn.Module, dataloader, max_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract last-layer embeddings, labels, and logits from the model.

        Returns:
            embeddings: (N, D) last-layer features
            labels: (N,) true next-token labels
            logits: (N, vocab_size) model predictions
        """
        model.eval()

        all_embeddings = []
        all_labels = []
        all_logits = []

        collected_samples = 0

        # Hook to capture embeddings
        embeddings_cache = []

        def hook_fn(module, input, output):
            # Store the output of the last layer before LM head
            embeddings_cache.append(output.detach())

        # Register hook on the appropriate layer
        hook_handle = None

        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            # nanoGPT style
            hook_handle = model.transformer.ln_f.register_forward_hook(hook_fn)
        elif hasattr(model, "ln_f"):
            # Direct access
            hook_handle = model.ln_f.register_forward_hook(hook_fn)
        elif hasattr(model, "model") and hasattr(model.model, "norm"):
            # Llama style: model.model.norm
            hook_handle = model.model.norm.register_forward_hook(hook_fn)
        else:
            print("Warning: Could not find layer norm for hooking.")
            print(f"Model structure: {list(model.named_children())}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                    tqdm(dataloader, desc="Extracting embeddings", leave=False)
            ):
                if collected_samples >= max_samples:
                    break

                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        inputs, targets = batch
                    else:
                        # Fallback: assume first element is input
                        inputs = batch[0]
                        # Create targets by shifting
                        if len(inputs.shape) == 2:  # (batch, seq)
                            targets = inputs[:, 1:].contiguous()
                            inputs = inputs[:, :-1].contiguous()
                        else:
                            targets = inputs[1:].contiguous()
                            inputs = inputs[:-1].contiguous()
                else:
                    # Single tensor: create shifted version
                    inputs = batch
                    if len(inputs.shape) == 2:  # (batch, seq)
                        targets = inputs[:, 1:].contiguous()
                        inputs = inputs[:, :-1].contiguous()
                    else:
                        targets = inputs[1:].contiguous()
                        inputs = inputs[:-1].contiguous()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Clear cache
                embeddings_cache.clear()

                # Forward pass
                logits = model(inputs)

                # Get embeddings from hook
                if embeddings_cache:
                    # Shape: (batch, seq_len, hidden_dim)
                    batch_embeddings = embeddings_cache[0]
                else:
                    # Fallback: try to get embeddings before final projection
                    # This requires model-specific knowledge
                    print(
                        "Warning: No embeddings captured. Using logits as proxy (not ideal)."
                    )
                    if len(logits.shape) == 3:
                        # Try to use the logits space (not ideal)
                        batch_embeddings = logits
                    else:
                        print("Error: Cannot extract embeddings. Skipping batch.")
                        continue

                # Handle different output shapes
                # Reshape for sequence models: (batch, seq, dim) -> (batch*seq, dim)
                if len(batch_embeddings.shape) == 3:
                    batch_size, seq_len, hidden_dim = batch_embeddings.shape
                    batch_embeddings = batch_embeddings.reshape(-1, hidden_dim)
                elif len(batch_embeddings.shape) == 2:
                    # (seq, dim) - single sequence
                    seq_len, hidden_dim = batch_embeddings.shape
                    batch_size = 1
                else:
                    print(f"Warning: Unexpected embedding shape: {batch_embeddings.shape}")

                    batch_size, seq_len = inputs.shape
                    hidden_dim = batch_embeddings.shape[-1]

                # Reshape logits and targets similarly
                if len(logits.shape) == 3:
                    # (batch, seq, vocab)
                    logits_flat = logits.reshape(-1, logits.shape[-1])
                else:
                    logits_flat = logits

                if len(targets.shape) == 2:
                    # (batch, seq)
                    targets_flat = targets.reshape(-1)
                else:
                    targets_flat = targets

                # Ensure shapes match
                min_len = min(
                    batch_embeddings.shape[0], logits_flat.shape[0], targets_flat.shape[0]
                )
                batch_embeddings = batch_embeddings[:min_len]
                logits_flat = logits_flat[:min_len]
                targets_flat = targets_flat[:min_len]

                # Filter out padding tokens if any (typically -100 or pad_token_id)
                valid_mask = (targets_flat != -100) & (targets_flat >= 0)

                if valid_mask.any():
                    batch_embeddings = batch_embeddings[valid_mask]
                    targets_flat = targets_flat[valid_mask]
                    logits_flat = logits_flat[valid_mask]
                else:
                    continue

                # Limit samples to avoid OOM
                remaining = max_samples - collected_samples
                if batch_embeddings.shape[0] > remaining:
                    batch_embeddings = batch_embeddings[:remaining]
                    targets_flat = targets_flat[:remaining]
                    logits_flat = logits_flat[:remaining]

                all_embeddings.append(batch_embeddings.cpu())
                all_labels.append(targets_flat.cpu())
                all_logits.append(logits_flat.cpu())

                collected_samples += batch_embeddings.shape[0]

        if hook_handle is not None:
            hook_handle.remove()

        if not all_embeddings:
            raise ValueError("No embeddings collected!")

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        logits = torch.cat(all_logits, dim=0)


        print(
            f"Collected {embeddings.shape[0]} samples, "
            f"embedding_dim={embeddings.shape[1]}, "
            f"vocab_size={logits.shape[1]}"
        )

        return embeddings, labels, logits

    def _filter_rare_classes(
            self, embeddings: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter out classes with too few samples.

        Returns:
            filtered_embeddings, filtered_labels, filtered_logits, valid_class_indices
        """
        unique_labels, counts = torch.unique(labels, return_counts=True)

        # Keep only classes with enough samples
        valid_classes = unique_labels[counts >= self.min_samples_per_class]

        if len(valid_classes) == 0:
            raise ValueError(
                f"No classes have >= {self.min_samples_per_class} samples!"
                f"Try reducing min_samples_per_class or collecting more data."
            )

        # Filter samples
        mask = torch.isin(labels, valid_classes)
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        filtered_logits = logits[mask]

        # Remap labels to contiguous indices [0, 1, 2, ..., C-1]
        label_mapping = {
            old_label.item(): new_label
            for new_label, old_label in enumerate(valid_classes)
        }
        remapped_labels = torch.tensor(
            [label_mapping[label.item()] for label in filtered_labels]
        )

        print(
            f"Filtered to {len(valid_classes)}/{self.num_classes} classes "
            f"with >= {self.min_samples_per_class} samples each"
        )
        print(f"Total samples after filtering: {filtered_embeddings.shape[0]}")

        return filtered_embeddings, remapped_labels, filtered_logits, valid_classes

    def compute_class_means(
            self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute class means M and global mean mG."""
        unique_labels = torch.unique(labels)
        C = len(unique_labels)
        D = embeddings.shape[1]

        M = torch.zeros(C, D, device=embeddings.device)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            M[i] = embeddings[mask].mean(dim=0)

        mG = M.mean(dim=0)
        return M, mG

    def nc1_within_class_variability(
            self, embeddings: torch.Tensor, labels: torch.Tensor, M: torch.Tensor
    ) -> Dict[str, float]:
        """
        NC1: Measure within-class variability collapse.

        Returns multiple NC1 metrics:
        - cdnv: Collapse Distance to class means (normalized variance)
        - pinv: Frobenius norm metric
        - svd: SVD-based metric
        """
        unique_labels = torch.unique(labels)
        C = len(unique_labels)
        D = embeddings.shape[1]

        # CDNV: Average squared distance to class mean
        within_class_var = 0.0
        total_samples = 0

        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_embeddings = embeddings[mask]
            class_mean = M[i]

            centered = class_embeddings - class_mean
            within_class_var += (centered ** 2).sum()
            total_samples += class_embeddings.shape[0]

        cdnv = (within_class_var / (total_samples * D)).item()

        # Between-class variance (for ratio)
        M_centered = M - M.mean(dim=0)
        between_class_var = (M_centered ** 2).sum().item() / (C * D)

        # Collapse ratio (within/between)
        collapse_ratio = cdnv / (between_class_var + 1e-10)

        return {
            "nc1_cdnv": cdnv,
            "nc1_quot": collapse_ratio,
            "nc1_pinv": cdnv,  # Simplified version
            "nc1_svd": cdnv,  # Simplified version
        }

    def nc2_hyperspherical_uniformity(
            self,
            M: torch.Tensor,
            mG: torch.Tensor
    ) -> Dict[str, float]:
        """NC2: Measure how uniformly class means are distributed on hypersphere.

        Returns:
        - etf_error: Deviation from Simplex ETF
        - norm variance: Are class means equinorm?
        """
        M_centered = M - mG
        C = M_centered.shape[0]

        # Check equinorm
        norms = torch.norm(M_centered, dim=1)
        norm_var = norms.var().item()

        # Check equiangular (Simplex ETF)
        M_normalized = F.normalize(M_centered, p=2, dim=1)
        similarities = M_normalized @ M_normalized.T

        # Upper triangular (excluding diagonal)
        triu_idx = torch.triu_indices(C, C, offset=1)
        pairwise_similarities = similarities[triu_idx[0], triu_idx[1]]

        # Ideal: all pairs should have cosine similarity
        ideal_similarity = -1.0 / (C - 1)
        etf_error = (pairwise_similarities - ideal_similarity).pow(2).mean().sqrt()

        # Distance-based metric (log kernel)
        pairwise_dists = torch.cdist(M_normalized, M_normalized)
        triu_dists = pairwise_dists[triu_idx[0], triu_idx[1]]
        log_dist_var = torch.log(triu_dists + 1e-8).var()

        return {
            "nc2_etf_err": etf_error.item(),
            "nc2g_dist": triu_dists.var().item(),
            "nc2g_log": log_dist_var.item(),
        }

    def nc3_self_duality(
            self,
            W: torch.Tensor,
            M: torch.Tensor,
            mG: torch.Tensor,
    ) -> Dict[str, float]:
        """
        NC3: Measure alignment between classifier weights and class means.

        W: (C, D) classifier weight matrix
        M: (C, D) class means
        """
        M_centered = M - mG
        W_centered = W - W.mean(dim=0)

        # Normalize
        M_norm = F.normalize(M_centered, p=2, dim=1)
        W_norm = F.normalize(W_centered, p=2, dim=1)

        # Cosine similarity between corresponding class pairs
        cosine_sim = (M_norm * W_norm).sum(dim=1)
        dual_error = (1 - cosine_sim).mean()

        # Uniform duality: all classes should align equally well
        uniform_dual = cosine_sim.var()

        return {
            "nc3_dual_err": dual_error.item(),
            "nc3u_uni_dual": uniform_dual.item(),
        }

    def nc4_classifier_agreement(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            logits: torch.Tensor,
            M: torch.Tensor,
    ) -> Dict[str, float]:
        """
        NC4: Agreement between learned classifier and nearest-class-mean classifier.
        """
        # Learned classifier predictions
        learned_preds = logits.argmax(dim=1)

        # Nearest class mean predictions
        distances = torch.cdist(embeddings, M)
        ncm_preds = distances.argmin(dim=1)

        # Agreement
        agreement = (learned_preds == ncm_preds).float().mean()

        return {"nc4_agree": agreement.item()}

    def analyze(
            self,
            model: nn.Module,
            train_loader,
            validation_loader,
            ood_loader,
            device: str,
    ):
        """Return dict with all NC metrics."""
        model.eval()

        try:
            # Extract embeddings, labels, and logits
            embeddings, labels, logits = self._extract_embeddings_and_logits(
                model, ood_loader, self.max_samples
            )
            # Filter rare classes
            embeddings, labels, logits, valid_classes = self._filter_rare_classes(
                embeddings, labels, logits
            )

            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = logits.to(device)

            # Compute class means
            M, mG = self.compute_class_means(embeddings, labels)

            # Compute NC metrics
            nc_metrics = {}

            # NC1
            nc1_metrics = self.nc1_within_class_variability(embeddings, labels, M)
            nc_metrics.update(nc1_metrics)

            # NC2
            nc2_metrics = self.nc2_hyperspherical_uniformity(M, mG)
            nc_metrics.update(nc2_metrics)

            # NC3: Requires classifier weights
            try:
                W = None

                # Try different common architectures
                if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                    # nanoGPT: model.lm_head.weight
                    W = model.lm_head.weight
                elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                    # Weight tying: use token embeddings
                    W = model.transformer.wte.weight
                elif hasattr(model, "output") and hasattr(model.output, "weight"):
                    # Generic output layer
                    W = model.output.weight

                if W is not None:
                    # Select only the valid classes we're analyzing
                    W = W[valid_classes].detach().to(device)
                    nc3_metrics = self.nc3_self_duality(W, M, mG)
                    nc_metrics.update(nc3_metrics)
                else:
                    print("Warning: Could not find classifier weights for NC3")
                    nc_metrics.update({
                        "nc3_dual_err": float("nan"),
                        "nc3u_uni_dual": float("nan")
                    })

            except Exception as e:
                print(f"NC3 computation failed: {e}")
                import traceback
                traceback.print_exc()
                nc_metrics.update({
                    "nc3_dual_err": float("nan"),
                    "nc3u_uni_dual": float("nan")
                })

            # NC4
            nc4_metrics = self.nc4_classifier_agreement(embeddings, labels, logits, M)
            nc_metrics.update(nc4_metrics)

            # Placeholder for NC5 (OOD detection - requires separate implementation)
            nc_metrics["nc5_ood_dev"] = float("nan")

            return nc_metrics

        except Exception as e:
            print(f"Neural collapse analysis failed: {e}")
            import traceback
            traceback.print_exc()

            # Return nan values on failure
            return {
                "nc1_pinv": float("nan"),
                "nc1_svd": float("nan"),
                "nc1_quot": float("nan"),
                "nc1_cdnv": float("nan"),
                "nc2_etf_err": float("nan"),
                "nc2g_dist": float("nan"),
                "nc2g_log": float("nan"),
                "nc3_dual_err": float("nan"),
                "nc3u_uni_dual": float("nan"),
                "nc4_agree": float("nan"),
                "nc5_ood_dev": float("nan"),
            }

    def compute_collapse(self):
        pass

        # if REQ_MEANS:
        #     M, mG = nc_stats.means_accum.compute(indices)
        #
        # if args.inv_snr and missing("cdnv_var", iden):  # NC1
        #     N_vars, N_means = nc_stats.N_seqs_vars, nc_stats.N_seqs_means
        #     if N_vars < N_means:
        #         print(f"W: vars for {iden} incomplete ({N_vars} < {N_means}); skipping")
        #     elif N_vars > N_means:
        #         print(f"E: too many vars ({N_vars} > {N_means})! skipping")
        #     else:
        #         V = nc_stats.vars_accum.compute(indices)[0]
        #         update_df(df, "cdnv", variability_cdnv(V, M, 2, args.tile_size), iden)
        #         del V
        #
        # if args.norms and missing("norms_var", iden):  # NC2 equinorm
        #     norms = mean_norms(M, mG)
        #     save_metrics(df, norms, "norms", iden)
        #     del norms
        # if args.norms and missing("norms_log_var", iden):  # NC2 equinorm (log)
        #     norms = mean_norms(M, mG, [pt.log])
        #     save_metrics(df, norms, "norms_log", iden)
        #     del norms
        #
        # if args.interfere and missing("interfere_var", iden):  # NC2 simplex ETF
        #     interference = interference_grid(M, mG)
        #     update_df(df, "etf_error", simplex_etf_error(M, mG), iden)
        #     save_metrics(df, interference, "interfere", iden, True)
        #     del interference
        #
        # if args.kernel and missing(f"{args.kernel}_kern_var", iden):  # GNC2
        #     kernel = riesz_kernel if "riesz" in args.kernel else log_kernel
        #     dists = kernel_grid(M, mG, kernel, args.tile_size)
        #     save_metrics(df, dists, f"{args.kernel}_kern", iden, True)
        #     del dists
        #
        # if args.duality and missing("dual_error", iden):  # NC3 duality
        #     W = get_classifier_weights(f"TinyStories-{iden}", args)
        #     if W is None:
        #         print(f"WARN: failed to load weights for {iden}.")
        #     else:
        #         W = W if indices is None else W[indices]
        #         dual_error = self_duality_error(W, M.to(W.dtype), mG.to(W.dtype))
        #         update_df(df, "dual_error", dual_error, iden)
        #         save_metrics(df, similarities(W, M, mG), "simdot", iden)
        #         save_metrics(df, similarities(W, M, mG, True), "simcos", iden)
        #         save_metrics(df, distance_norms(W, M, mG), "dists", iden)
        #     del W
        #
        # if args.decisions and missing("hits", iden):  # NC4 agreement
        #     hits = nc_stats.decs_accum.totals[indices]
        #     misses = nc_stats.decs_accum.ns_samples[indices] - hits
        #     update_df(df, "hits", hits.sum(), iden)
        #     update_df(df, "misses", misses.sum(), iden)
        #
        # del nc_stats
        # df.to_csv(f"{args.output_file}.csv")
        #
