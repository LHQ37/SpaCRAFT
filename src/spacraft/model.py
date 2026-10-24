import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from .networks import *
from .loss_functions import *
from .banksy_features import BanksyConfig, build_banksy_features


class BanksyProjection(nn.Module):
    """Configurable BANKSY-only projection; never constructed on Raw paths."""

    def __init__(self, input_dim, output_dim, layers=1, hidden_dim=None,
                 activation="identity", normalization="none", dropout=0.0,
                 bias=True, pre_layernorm=False, post_layernorm=False):
        super().__init__()
        if layers not in {1, 2}:
            raise ValueError("projection layers must be 1 or 2")
        if activation not in {"identity", "relu", "gelu"}:
            raise ValueError("unknown projection activation")
        if normalization not in {"none", "layernorm"}:
            raise ValueError("unknown projection normalization")
        modules = []
        if pre_layernorm:
            modules.append(nn.LayerNorm(input_dim))
        if layers == 1:
            modules.append(nn.Linear(input_dim, output_dim, bias=bias))
            if normalization == "layernorm":
                modules.append(nn.LayerNorm(output_dim))
            if dropout:
                modules.append(nn.Dropout(dropout))
        else:
            hidden_dim = output_dim if hidden_dim is None else hidden_dim
            modules.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            if normalization == "layernorm":
                modules.append(nn.LayerNorm(hidden_dim))
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "gelu":
                modules.append(nn.GELU())
            if dropout:
                modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        if post_layernorm:
            modules.append(nn.LayerNorm(output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, values):
        return self.network(values)


class SpaCRAFTModel:
    """GAT-based SpaCRAFT representation model."""

    def __init__(
        self,
        adata_list,
        adata_concat,
        input_mat_key="X_pca",
        input_dim=100,
        hidden_dims_G=[50],
        decoder_hidden_dims=None,
        latent_dim=30,
        hidden_dims_D=[50],
        lambda_adv=1,
        lambda_rec=10,
        lambda_cluster_contrast=2.0,
        enable_contrastive=True,
        seed=1236,
        learn_rates=[1e-3, 5e-4],
        steps=500,
        multi_slice_method="raw",
        banksy_kgeom=15,
        lambda_banksy=0.2,
        banksy_mu=1.5,
        banksy_agf_multiplier=2.0,
        banksy_gaussian_scale_fraction=0.5,
        banksy_normalization_strategy="per_slice_component_zscore",
        projection_output_dim=None,
        projection_layers=1,
        projection_hidden_dim=None,
        projection_activation="identity",
        projection_normalization="none",
        projection_dropout=0.0,
        projection_bias=True,
        optimizer_type="adam",
        weight_decay=0.0,
        optimizer_betas=(0.9, 0.999),
        optimizer_eps=1e-8,
        scheduler_type="none",
        scheduler_eta_min_ratio=0.01,
        projection_lr_multiplier=1.0,
        gradient_clip_max_norm=None,
        contrastive_temperature=0.5,
        encoder_feature_dropout=0.0,
        attention_dropout=0.0,
        gat_normalization="none",
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.lambda_adv = lambda_adv
        self.lambda_rec = lambda_rec
        self.lambda_cluster_contrast = lambda_cluster_contrast
        self.enable_contrastive = enable_contrastive

        self.latent_dim = latent_dim
        self.contrastive_temperature = contrastive_temperature
        self.encoder_feature_dropout = float(encoder_feature_dropout)
        self.attention_dropout = float(attention_dropout)
        self.gat_normalization = gat_normalization.lower()
        self.device = device
        self.n_cls = len(adata_list)
        self.n_spots = adata_concat.shape[0]
        self.steps = int(steps)
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        self.optimizer_type = optimizer_type.lower()
        self.weight_decay = float(weight_decay)
        self.optimizer_betas = tuple(optimizer_betas)
        self.optimizer_eps = float(optimizer_eps)
        self.scheduler_type = scheduler_type.lower()
        self.scheduler_eta_min_ratio = float(scheduler_eta_min_ratio)
        self.projection_lr_multiplier = float(projection_lr_multiplier)
        self.gradient_clip_max_norm = gradient_clip_max_norm
        if self.optimizer_type not in {"adam", "adamw"}:
            raise ValueError("optimizer_type must be 'adam' or 'adamw'")
        if self.scheduler_type not in {"none", "cosine_decay"}:
            raise ValueError("scheduler_type must be 'none' or 'cosine_decay'")
        if multi_slice_method not in {"raw", "banksy"}:
            raise ValueError("multi_slice_method must be 'raw' or 'banksy'")
        self.multi_slice_method = (
            "raw" if self.n_cls < 2 else multi_slice_method
        )

        self.slice_index = np.zeros((self.n_spots,))
        self.indices = [0]
        idx = 0
        for i in range(self.n_cls):
            idx += adata_list[i].shape[0]
            self.indices.append(idx)
            self.slice_index[self.indices[-2]:self.indices[-1]] = i

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        if self.multi_slice_method == "banksy":
            banksy_matrix, self.banksy_audit = build_banksy_features(
                adata_list, adata_concat, input_mat_key,
                BanksyConfig(kgeom=banksy_kgeom,
                             lambda_banksy=lambda_banksy, mu=banksy_mu,
                             agf_multiplier=banksy_agf_multiplier,
                             gaussian_scale_fraction=banksy_gaussian_scale_fraction,
                             normalization_strategy=banksy_normalization_strategy))
            self.banksy_features = torch.from_numpy(banksy_matrix).float().to(self.device)
            model_input_dim = (
                input_dim if projection_output_dim is None
                else int(projection_output_dim))
            default_projection = (
                projection_layers == 1
                and model_input_dim == input_dim
                and projection_normalization == "none"
                and float(projection_dropout) == 0.0
                and bool(projection_bias)
                and banksy_normalization_strategy
                == "per_slice_component_zscore")
            if default_projection:
                self.banksy_projection = nn.Linear(
                    banksy_matrix.shape[1], input_dim, bias=True).to(self.device)
            else:
                self.banksy_projection = BanksyProjection(
                    banksy_matrix.shape[1], model_input_dim,
                    layers=projection_layers,
                    hidden_dim=projection_hidden_dim,
                    activation=projection_activation.lower(),
                    normalization=projection_normalization.lower(),
                    dropout=projection_dropout, bias=projection_bias,
                    pre_layernorm=(
                        banksy_normalization_strategy
                        == "per_slice_component_zscore_plus_post_concat_layernorm"),
                    post_layernorm=(
                        banksy_normalization_strategy
                        == "per_slice_component_zscore_plus_post_projection_layernorm"),
                ).to(self.device)
            self.model_input_dim = model_input_dim
            self.banksy_audit["dimensions"] = {
                "raw_input": input_dim,
                "fused_before_projection": int(banksy_matrix.shape[1]),
                "projected_model_input": model_input_dim,
            }
            self.banksy_audit["projection_parameters"] = sum(
                p.numel() for p in self.banksy_projection.parameters())
        model_input_dim = (
            self.model_input_dim
            if self.multi_slice_method == "banksy" else input_dim)
        self.encoder = GATEncoder(
            model_input_dim, hidden_dims_G, latent_dim,
            feature_dropout=self.encoder_feature_dropout,
            attention_dropout=self.attention_dropout,
            normalization=self.gat_normalization).to(self.device)
        decoder_hidden_dims = (
            hidden_dims_G if decoder_hidden_dims is None
            else decoder_hidden_dims)
        self.decoder = MLPDecoder(
            latent_dim, decoder_hidden_dims, model_input_dim).to(self.device)
        self.discriminator = Discriminator(
            model_input_dim, hidden_dims_D).to(self.device)

        self.params_G = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + (list(self.banksy_projection.parameters())
               if self.multi_slice_method == "banksy" else [])
        )
        self.params_D = list(self.discriminator.parameters())
        optimizer_class = (
            optim.Adam if self.optimizer_type == "adam" else optim.AdamW)
        if (self.multi_slice_method == "banksy"
                and self.projection_lr_multiplier != 1.0):
            projection_ids = {
                id(parameter)
                for parameter in self.banksy_projection.parameters()}
            generator_groups = [
                {"params": [parameter for parameter in self.params_G
                            if id(parameter) not in projection_ids],
                 "lr": learn_rates[0]},
                {"params": list(self.banksy_projection.parameters()),
                 "lr": learn_rates[0] * self.projection_lr_multiplier},
            ]
        else:
            generator_groups = self.params_G
        optimizer_kwargs = {
            "weight_decay": self.weight_decay,
            "betas": self.optimizer_betas,
            "eps": self.optimizer_eps,
        }
        self.optimizer_G = optimizer_class(
            generator_groups, lr=learn_rates[0], **optimizer_kwargs)
        self.optimizer_D = optimizer_class(
            self.params_D, lr=learn_rates[1], **optimizer_kwargs)
        self.scheduler_G = None
        self.scheduler_D = None
        if self.scheduler_type == "cosine_decay":
            t_max = max(1, self.steps)
            self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_G, T_max=t_max,
                eta_min=learn_rates[0] * self.scheduler_eta_min_ratio)
            self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_D, T_max=t_max,
                eta_min=learn_rates[1] * self.scheduler_eta_min_ratio)

        self.X_raw = torch.from_numpy(
            np.asarray(adata_concat.obsm[input_mat_key])).float().to(self.device)
        self.X = self._model_input()
        if "graph_list" in adata_concat.uns.keys():
            self.G = np.zeros(
                (adata_concat.X.shape[0], adata_concat.X.shape[0]), dtype=float
            )
            for i in range(len(adata_list)):
                subgraph = adata_concat.uns["graph_list"][i].copy().astype(float)
                self.G[
                    self.indices[i]:self.indices[i + 1],
                    self.indices[i]:self.indices[i + 1],
                ] = subgraph
            self.G = torch.from_numpy(self.G).float().to(self.device)
        elif "graph" in adata_concat.obsm.keys():
            self.G = torch.from_numpy(
                adata_concat.obsm["graph"].copy()
            ).float().to(self.device)
        else:
            raise ValueError("No graph is provided !")

        if self.enable_contrastive:
            graph_neigh_np = self.G.cpu().numpy() if isinstance(self.G, torch.Tensor) else self.G
            n_nodes = graph_neigh_np.shape[0]
            graph_neigh_np = graph_neigh_np + np.eye(n_nodes)
            graph_neigh_np = np.where(graph_neigh_np > 1, 1, graph_neigh_np)
            self.graph_neigh = torch.from_numpy(graph_neigh_np).float().to(self.device)
        else:
            self.graph_neigh = None
            self.label_CSL = None

    def _model_input(self):
        if self.multi_slice_method == "banksy":
            return self.banksy_projection(self.banksy_features)
        return self.X_raw

    def encode(self):
        self.X = self._model_input()
        return self.encoder(self.X, self.G)

    def train(self, report_loss=True, report_interval=100,
              record_loss_history=False, record_gradient_history=False,
              step_offset=0, record_loss_gradient_diagnostics=False,
              loss_diagnostic_steps=None):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        if self.multi_slice_method == "banksy":
            self.banksy_projection.train()
        if record_loss_history:
            self.loss_history = []
        if record_gradient_history:
            self.gradient_history = []
            self.learning_rate_history = []
            projection_initial = (
                torch.cat([parameter.detach().flatten().cpu()
                           for parameter in self.banksy_projection.parameters()])
                if self.multi_slice_method == "banksy" else None)
            encoder_initial = torch.cat([
                parameter.detach().flatten().cpu()
                for parameter in self.encoder.parameters()])
            decoder_initial = torch.cat([
                parameter.detach().flatten().cpu()
                for parameter in self.decoder.parameters()])
        if record_loss_gradient_diagnostics:
            self.loss_gradient_diagnostics = []
            if loss_diagnostic_steps is None:
                loss_diagnostic_steps = {0, 10, 20, 50, 100, 200, 300, 400,
                                         max(0, self.steps - 1)}
            else:
                loss_diagnostic_steps = set(loss_diagnostic_steps)

        if self.steps:
            print("Start training the autoencoder and the discriminator ...\n")

        for step in range(self.steps):
            global_step = int(step_offset) + step
            self.X = self._model_input()
            Z_X = self.encoder(self.X, self.G)
            X_rec = self.decoder(Z_X)

            X_src = self.discriminator(self.X)
            X_rec_src = self.discriminator(X_rec)

            loss_adv_g = adversarial_loss_g(X_rec_src)
            loss_adv_d = adversarial_loss_d(X_src, X_rec_src)
            loss_rec = mse_loss(self.X, X_rec)

            if self.enable_contrastive > 0:
                loss_cluster_contrast = contrastive_clustering_loss(
                    Z_X, self.graph_neigh,
                    temperature=self.contrastive_temperature
                )
            else:
                loss_cluster_contrast = torch.tensor(0.0).to(self.device)

            loss_d = self.lambda_adv * loss_adv_d
            self.optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)

            loss_g = (
                self.lambda_adv * loss_adv_g
                + self.lambda_rec * loss_rec
                + self.lambda_cluster_contrast * loss_cluster_contrast
            )
            if (record_loss_gradient_diagnostics
                    and step in loss_diagnostic_steps):
                diagnostic_step = (
                    self.steps if step == self.steps - 1 else global_step)
                module_parameters = {
                    "projection": (
                        list(self.banksy_projection.parameters())
                        if self.multi_slice_method == "banksy" else []),
                    "encoder": list(self.encoder.parameters()),
                    "decoder": list(self.decoder.parameters()),
                }
                weighted_losses = {
                    "adversarial_generator":
                        self.lambda_adv * loss_adv_g,
                    "reconstruction": self.lambda_rec * loss_rec,
                    "cluster_contrast":
                        self.lambda_cluster_contrast * loss_cluster_contrast,
                }

                def flattened_gradient(value, parameters):
                    if not parameters or not value.requires_grad:
                        return torch.empty(0, device=self.device)
                    gradients = torch.autograd.grad(
                        value, parameters, retain_graph=True,
                        allow_unused=True)
                    return torch.cat([
                        (gradient if gradient is not None
                         else torch.zeros_like(parameter)).reshape(-1)
                        for parameter, gradient in zip(parameters, gradients)
                    ])

                for module_name, parameters in module_parameters.items():
                    total_gradient = flattened_gradient(loss_g, parameters)
                    reconstruction_gradient = flattened_gradient(
                        weighted_losses["reconstruction"], parameters)
                    total_norm = float(total_gradient.norm().detach().cpu())
                    for loss_name, weighted_loss in weighted_losses.items():
                        gradient = flattened_gradient(weighted_loss, parameters)
                        norm = float(gradient.norm().detach().cpu())

                        def cosine(left, right):
                            denominator = left.norm() * right.norm()
                            if float(denominator.detach().cpu()) == 0:
                                return float("nan")
                            return float(
                                torch.dot(left, right).div(denominator)
                                .detach().cpu())

                        self.loss_gradient_diagnostics.append({
                            "step": int(diagnostic_step),
                            "loss_name": loss_name,
                            "module": module_name,
                            "raw_loss": float({
                                "adversarial_generator": loss_adv_g,
                                "reconstruction": loss_rec,
                                "cluster_contrast": loss_cluster_contrast,
                            }[loss_name].detach().cpu()),
                            "weight": float({
                                "adversarial_generator": self.lambda_adv,
                                "reconstruction": self.lambda_rec,
                                "cluster_contrast":
                                    self.lambda_cluster_contrast,
                            }[loss_name]),
                            "weighted_loss": float(
                                weighted_loss.detach().cpu()),
                            "weighted_loss_ratio": float(
                                weighted_loss.detach().cpu()
                                / (loss_g.detach().cpu().abs() + 1e-12)),
                            "gradient_norm": norm,
                            "total_gradient_norm": total_norm,
                            "gradient_ratio": (
                                norm / total_norm if total_norm else 0.0),
                            "cosine_vs_reconstruction": cosine(
                                gradient, reconstruction_gradient),
                            "cosine_vs_total": cosine(
                                gradient, total_gradient),
                            "gradient_is_zero": norm == 0.0,
                            "gradient_is_finite": bool(
                                torch.isfinite(gradient).all().detach().cpu()),
                        })
            self.optimizer_G.zero_grad()
            loss_g.backward()

            if record_gradient_history:
                def grad_norm(parameters):
                    values = [
                        parameter.grad.detach().norm().cpu()
                        for parameter in parameters if parameter.grad is not None]
                    return float(torch.stack(values).norm()) if values else 0.0

                projection_parameters = (
                    list(self.banksy_projection.parameters())
                    if self.multi_slice_method == "banksy" else [])
                projection_current = (
                    torch.cat([parameter.detach().flatten().cpu()
                               for parameter in projection_parameters])
                    if projection_parameters else None)
                self.gradient_history.append({
                    "step": global_step,
                    "projection_gradient_norm": grad_norm(projection_parameters),
                    "encoder_gradient_norm": grad_norm(self.encoder.parameters()),
                    "decoder_gradient_norm": grad_norm(self.decoder.parameters()),
                    "discriminator_gradient_norm": grad_norm(
                        self.discriminator.parameters()),
                    "total_gradient_norm": grad_norm(
                        list(self.params_G) + list(self.params_D)),
                    "max_single_parameter_gradient": max(
                        [float(parameter.grad.detach().abs().max().cpu())
                         for parameter in list(self.params_G) + list(self.params_D)
                         if parameter.grad is not None] or [0.0]),
                    "projection_parameter_change_norm": (
                        float((projection_current - projection_initial).norm())
                        if projection_current is not None else 0.0),
                    "encoder_parameter_change_norm": float((
                        torch.cat([
                            parameter.detach().flatten().cpu()
                            for parameter in self.encoder.parameters()])
                        - encoder_initial).norm()),
                    "decoder_parameter_change_norm": float((
                        torch.cat([
                            parameter.detach().flatten().cpu()
                            for parameter in self.decoder.parameters()])
                        - decoder_initial).norm()),
                    "projection_weight_norm": float(torch.cat([
                        parameter.detach().flatten().cpu()
                        for parameter in projection_parameters]).norm())
                    if projection_parameters else 0.0,
                    "encoder_weight_norm": float(torch.cat([
                        parameter.detach().flatten().cpu()
                        for parameter in self.encoder.parameters()]).norm()),
                })

            if self.gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.params_G, float(self.gradient_clip_max_norm))
                torch.nn.utils.clip_grad_norm_(
                    self.params_D, float(self.gradient_clip_max_norm))
            if record_gradient_history:
                self.learning_rate_history.append({
                    "step": global_step,
                    "generator_learning_rate":
                    float(self.optimizer_G.param_groups[0]["lr"]),
                    "projection_learning_rate": float(
                        self.optimizer_G.param_groups[-1]["lr"]),
                    "discriminator_learning_rate":
                    float(self.optimizer_D.param_groups[0]["lr"]),
                })
            self.optimizer_D.step()
            self.optimizer_G.step()
            if self.scheduler_D is not None:
                self.scheduler_D.step()
            if self.scheduler_G is not None:
                self.scheduler_G.step()
            if record_loss_history:
                self.loss_history.append({
                    "step": global_step,
                    "base_loss": float(loss_g.detach().cpu()),
                    "total_generator_loss": float(loss_g.detach().cpu()),
                    "discriminator_loss": float(loss_d.detach().cpu()),
                    "loss_adv_g": float(loss_adv_g.detach().cpu()),
                    "loss_adv_d": float(loss_adv_d.detach().cpu()),
                    "loss_rec": float(loss_rec.detach().cpu()),
                    "loss_cluster_contrast": float(
                        loss_cluster_contrast.detach().cpu()),
                    "weighted_loss_adv_g": float(
                        (self.lambda_adv * loss_adv_g).detach().cpu()),
                    "weighted_loss_rec": float(
                        (self.lambda_rec * loss_rec).detach().cpu()),
                    "weighted_loss_cluster_contrast": float(
                        (self.lambda_cluster_contrast
                         * loss_cluster_contrast).detach().cpu()),
                })

            if report_loss and ((step + 1) % report_interval == 0 or step == 0):
                print("Step: %s" % step)
                print("loss_G: %.4f, loss_D: %.4f" % (loss_g.item(), loss_d.item()))
                if self.enable_contrastive:
                    print(
                        "loss_adv_g: %.4f, loss_adv_d: %.4f, loss_rec: %.4f, "
                        "loss_cluster_contrast: %.4f"
                        % (
                            loss_adv_g.item(),
                            loss_adv_d.item(),
                            loss_rec.item(),
                            loss_cluster_contrast.item(),
                        )
                    )
                else:
                    print(
                        "loss_adv_g: %.4f, loss_adv_d: %.4f, loss_rec: %.4f"
                        % (loss_adv_g.item(), loss_adv_d.item(), loss_rec.item())
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.steps:
            print("Done !\n")


        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        print("Takes %.2f seconds in total\n" % (end_time - begin_time))

    def eval(
        self,
        adata_list,
        slice_names=None,
        save_dir_data=None,
        save=False,
    ):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        if self.multi_slice_method == "banksy":
            self.banksy_projection.eval()

        with torch.no_grad():
            self.X = self._model_input()
            Z_X = self.encoder(self.X, self.G)
        Z_X_np = Z_X.cpu().numpy()

        for i in range(len(adata_list)):
            start_idx, end_idx = self.indices[i], self.indices[i + 1]
            adata_list[i].obsm["spacraft_latent"] = Z_X_np[start_idx:end_idx]
            adata_list[i].obsm["spacraft_latent_encoder"] = Z_X_np[start_idx:end_idx]

            if save:
                if slice_names is None:
                    raise ValueError("slice_names must be provided when save=True")
                if save_dir_data is None:
                    raise ValueError("save_dir_data must be provided when save=True")
                os.makedirs(save_dir_data, exist_ok=True)
                filename = (
                    "spacraft_"
                    + slice_names[i]
                    + f"_{self.steps}.h5ad"
                )
                adata_list[i].write_h5ad(os.path.join(save_dir_data, filename))
