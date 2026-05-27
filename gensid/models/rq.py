import torch
import torch.nn as nn

from gensid.models.vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta=1,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_ini(self, x, use_sk):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            x_res = quantizer.vq_init(residual, use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x, labels, use_sk=True):
        all_losses = []
        all_indices = []
        all_quantized = []
        all_res = []

        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            # NOTE part of slow loss
            # label = labels[str(idx)]
            label = None

            x_res, loss, indices, x_res_codebook = quantizer(residual, label, idx, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_quantized.append(x_res_codebook)
            all_res.append(x_res)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_quantized = torch.stack(all_quantized, dim=1)
        all_res = torch.stack(all_res, dim=1)

        return x_q, mean_losses, all_indices, all_quantized, all_res
