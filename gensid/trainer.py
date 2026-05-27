import logging
import json
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from gensid.utils import ensure_dir, set_color, get_local_time
import os
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.use_sk = not args.disable_sk
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        if args.saved_model_dir is not None:
            saved_model_dir = args.saved_model_dir
        else:
            saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": []}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        if self.args.compile:
            model = torch.compile(model)
        self.trained_loss = {"total": [], "rqvae": [], "recon": [], "cf": []}
        self.valid_collision_rate = {"val": []}
        self.graph_encoder = args.graph_encoder
        self.batch_size = args.batch_size

        self.plain_reconstruction = args.plain_reconstruction
        if self.plain_reconstruction:
            logging.info("PLAIN SEMANTIC RECONSTRUCTION set")

        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, "tensorboard"))

        self.max_cluster_level = args.max_cluster_level
        if self.max_cluster_level is None:
            self.max_cluster_level = len(args.num_emb_list) - 1  # last one reserved for edge reconstruction loss

    def _build_optimizer(self):

        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device, non_blocking=True)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning("Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _calculate_unused_codebooks(self, used_codebooks):
        unused_codebooks = []
        for idx, layer in enumerate(self.model.rq.vq_layers):
            num_total_codebooks = layer.n_e
            num_used_codebooks = len(used_codebooks[idx])
            unused_codebooks.append(num_total_codebooks - num_used_codebooks)
        return sum(unused_codebooks)

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained

        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=n_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels

    def vq_init(self, dataset):
        if getattr(self.args, "disable_vq_init", True):
            self.logger.info(set_color("Skip vq initialization", "yellow") + ": args.vq_init=False")
            return
        self.model.eval()
        init_loader = DataLoader(
            dataset, num_workers=self.args.num_workers, batch_size=len(dataset), shuffle=True, pin_memory=True
        )
        print(len(init_loader))
        iter_data = tqdm(
            init_loader,
            total=len(init_loader),
            ncols=100,
            desc=set_color("Initialization of vq", "pink"),
        )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device, non_blocking=True)

            self.model.vq_initialization(data, self.use_sk)

    def _train_epoch(self, train_data, epoch_idx, pyg_data=None):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        used_codebooks = [set() for _ in self.model.rq.vq_layers]

        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )

        self.labels = None
        cluster_labels = False

        for batch_idx, _data in enumerate(iter_data):
            spectral_vectors = None
            edges_weights = None
            if self.graph_encoder:
                emb_idx = _data.n_id[: _data.batch_size]
                graph_data = _data.to(self.device, non_blocking=True)
                data = _data.x[: _data.batch_size]
                recon_tgt = data
            elif self.plain_reconstruction:
                data, emb_idx, recon_tgt = _data[0], _data[1], _data[2]
                data = data.to(self.device, non_blocking=True)
                recon_tgt = recon_tgt.to(self.device, non_blocking=True)
            elif self.args.spectral_dim > 0:
                data, emb_idx, recon_tgt, spectral_vectors = _data[0], _data[1], _data[2], _data[3]
                data = data.to(self.device, non_blocking=True)
                recon_tgt = recon_tgt.to(self.device, non_blocking=True)
                spectral_vectors = spectral_vectors.to(self.device, non_blocking=True)
            elif self.args.use_edge_reconstruction_loss and self.args.cluster_labels_path is not None:
                data, emb_idx, recon_tgt, cluster_labels, edges_weights = (
                    _data[0],
                    _data[1],
                    _data[2],
                    _data[3],
                    _data[4],
                )
                data = data.to(self.device, non_blocking=True)
                recon_tgt = recon_tgt.to(self.device, non_blocking=True)
                cluster_labels = cluster_labels[:, : self.max_cluster_level]
                cluster_labels = cluster_labels.to(self.device, non_blocking=True)
            elif self.args.use_edge_reconstruction_loss:
                data, emb_idx, _, edges_weights = _data
                data = data.to(self.device, non_blocking=True)
                recon_tgt = data
            else:
                data, emb_idx = _data[0], _data[1]
                data = data.to(self.device, non_blocking=True)
                recon_tgt = data

            self.optimizer.zero_grad()

            if self.graph_encoder:
                out, rq_loss, indices, dense_out, dense_out_codebooks, x_res_list = self.model(
                    data, graph_data, self.labels, self.use_sk
                )
            else:
                out, rq_loss, indices, dense_out, dense_out_codebooks, x_res_list = self.model(data, self.labels)

            for codebook_idx in range(indices.shape[-1]):
                used_codebooks[codebook_idx].update(indices[:, codebook_idx].detach().cpu().tolist())

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(
                out,
                rq_loss,
                emb_idx,
                dense_out,
                dense_out_codebooks=dense_out_codebooks,
                x_res_list=x_res_list,
                xs=recon_tgt,
                edges_weights=edges_weights,
                spectral_vectors=spectral_vectors,
                cluster_labels=cluster_labels,
            )
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += cf_loss.item() if cf_loss != 0 else cf_loss
            total_quant_loss += quant_loss.item()

        unused_codebooks = self._calculate_unused_codebooks(used_codebooks)

        return total_loss, total_recon_loss, total_cf_loss, quant_loss.item(), unused_codebooks

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate   ", "pink"),
        )
        indices_set = set()

        num_sample = 0

        self.labels = None
        for batch_idx, _data in enumerate(iter_data):
            if self.graph_encoder:
                emb_idx = _data.n_id[: _data.batch_size]
                graph_data = _data.to(self.device, non_blocking=True)
                data = _data.x[: _data.batch_size]
            else:
                data, emb_idx = _data[0], _data[1]
            num_sample += len(data)
            data = data.to(self.device, non_blocking=True)
            if self.graph_encoder:
                indices = self.model.get_indices(data, graph_data, self.labels)
            else:
                indices = self.model.get_indices(data, self.labels)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set)) / num_sample

        return collision_rate

    def _set_trained_flag(self, ckpt_file: str):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)
        metrics_path = ckpt_path.replace(".pth", ".json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["fully_trained"] = True
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(self.ckpt_dir, "epoch_%d_collision_%.4f_model.pth" % (epoch, collision_rate))
        )
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        metrics_path = ckpt_path.replace(".pth", ".json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "best_loss": float(self.best_loss),
                    "best_collision_rate": float(self.best_collision_rate),
                    "ckpt_path": ckpt_path,
                },
                f,
                indent=2,
            )

        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, cf_loss, unused_codebooks):
        self.tb_writer.add_scalar("train/epoch_time_sec", e_time - s_time, epoch_idx)
        self.tb_writer.add_scalar("train/train_loss", loss, epoch_idx)
        self.tb_writer.add_scalar("train/loss_reconstruction", recon_loss, epoch_idx)
        self.tb_writer.add_scalar("train/loss_cf", cf_loss, epoch_idx)
        self.tb_writer.add_scalar("train/unused_codebooks", unused_codebooks, epoch_idx)

        train_loss_output = (
            set_color("epoch %d training", "green") + " [" + set_color("time", "blue") + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output += ", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output += ", "
        train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        train_loss_output += ", "
        train_loss_output += set_color("unused codebooks", "blue") + f": {unused_codebooks}"
        return train_loss_output + "]"

    def fit(self, dataset, data):

        cur_eval_step = 0
        self.vq_init(dataset)
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            pyg_data = None
            if self.args.use_edge_reconstruction_loss:
                pyg_data = dataset.pyg_data
            train_loss, train_recon_loss, cf_loss, quant_loss, unused_codebooks = self._train_epoch(
                data, epoch_idx, pyg_data
            )

            training_end_time = time()

            train_loss_output = self._generate_train_loss_output(
                epoch_idx,
                training_start_time,
                training_end_time,
                train_loss,
                train_recon_loss,
                cf_loss,
                unused_codebooks,
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate, ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)
                self.tb_writer.add_scalar("train/collision_rate", collision_rate, epoch_idx)

                self.logger.info(valid_score_output)

        self._set_trained_flag(ckpt_file=self.best_collision_ckpt)
        return self.best_loss, self.best_collision_rate
