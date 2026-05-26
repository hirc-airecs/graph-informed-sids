import os
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import BaseSIDTokenizer, EncoderDecoderTokenizerMixin
from genrec.models.TIGER.layers import RQVAEModel
from genrec.utils import list_to_str


class TIGERTokenizer(EncoderDecoderTokenizerMixin, BaseSIDTokenizer):
    """
    Tokenizer for the TIGER model.

    An example when "rq_codebook_size == 256, rq_n_codebooks == 3, n_user_tokens == 2000":
        0: padding
        1-256: digit 1
        257-512: digit 2
        513-768: digit 3
        769-1024: digit 4 (used to avoid conflicts)
        1025-3024: user tokens
        3025: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """
    def _get_index_factory(self) -> str:
        sid_type = "faiss" if self.config['rq_faiss'] else "rqvae"
        return f'{os.path.basename(self.config["sent_emb_model"])}_{list_to_str(self.codebook_sizes, remove_blank=True)}_{sid_type}.sem_ids'

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        sent_emb_model = SentenceTransformer(
            self.config['sent_emb_model'],
        ).to(self.config['device'])

        if 'sent_emb_max_length' in self.config:
            sent_emb_model.max_seq_length = self.config['sent_emb_max_length']

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
        )

        sent_embs.tofile(output_path)
        return sent_embs

    def _train_rqvae(self, sent_embs: torch.Tensor, model_path: str) -> RQVAEModel:
        """
        Trains the RQ-VAE model using the given sentence embeddings.

        Args:
            sent_embs (torch.Tensor): Array of sentence embeddings.
            model_path (str): Path to save the trained model.

        Returns:
            rqvae_model: Trained RQ-VAE model.
        """
        device = self.config['device']

        # Initialize RQ-VAE model
        all_hidden_sizes = [sent_embs.shape[1]] + self.config['rqvae_hidden_sizes']
        rqvae_model = RQVAEModel(
            hidden_sizes=all_hidden_sizes,
            n_codebooks=self.config['rq_n_codebooks'],
            codebook_size=self.config['rq_codebook_size'],
            dropout=self.config['rqvae_dropout'],
            low_usage_threshold=self.config['rqvae_low_usage_threshold']
        ).to(device)
        self.log(rqvae_model)
        if os.path.exists(model_path):
            self.log(f"[TOKENIZER] Loading RQ-VAE model from {model_path}...")
            rqvae_model.load_state_dict(torch.load(model_path, weights_only=False))
            return rqvae_model

        # Model training
        batch_size = self.config['rqvae_batch_size']
        num_epochs = self.config['rqvae_epoch']
        beta = self.config['rqvae_beta']
        verbose = self.config['rqvae_verbose']

        rqvae_model.generate_codebook(sent_embs, device)
        optim_name = self.config.get("rqvae_optim", "adagrad")
        if optim_name == "adagrad":
            optimizer = torch.optim.Adagrad(rqvae_model.parameters(), lr=self.config['rqvae_lr'])
        elif optim_name == "adamw":
            optimizer = torch.optim.AdamW(rqvae_model.parameters(), lr=self.config['rqvae_lr'], weight_decay=self.config.get('rqvae_weight_decay', 0))
        else:
            raise ValueError("rqvae_optim can only be adagrad or adamw!")
        train_dataset = TensorDataset(sent_embs)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.log("[TOKENIZER] Training RQ-VAE model...")
        rqvae_model.train()
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0.0
            total_rec_loss = 0.0
            total_quant_loss = 0.0
            total_count = 0
            for batch in dataloader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon_x, quant_loss, count = rqvae_model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                loss = reconstruction_mse_loss + beta * quant_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().item()
                total_rec_loss += reconstruction_mse_loss.detach().cpu().item()
                total_quant_loss += quant_loss.detach().cpu().item()
                total_count += count

            if (epoch + 1) % verbose == 0:
                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {total_loss/ len(dataloader)}\n"
                    f"\t  Unused codebook: {total_count/ len(dataloader)}\n"
                    f"\t  Reconstruction loss: {total_rec_loss/ len(dataloader)}\n"
                    f"\t  Quantization loss: {total_quant_loss/ len(dataloader)}\n")
        self.log("[TOKENIZER] RQ-VAE training complete.")

        # Save model
        torch.save(rqvae_model.state_dict(), model_path, pickle_protocol=4)
        return rqvae_model

    def _extend_semantic_ids(self, sem_ids: np.ndarray):
        """
        Extends the semantic IDs from k digits to (k + 1) digits to avoid conflict.

        Args:
            sem_ids (np.ndarray): The input array of semantic IDs.

        Returns:
            dict: A dictionary mapping item IDs to semantic IDs.
        """
        sem_id2item = defaultdict(list)
        item2sem_ids = {}
        max_conflict = 0
        for i in range(sem_ids.shape[0]):
            str_id = ' '.join(map(str, sem_ids[i].tolist()))
            sem_id2item[str_id].append(i + 1)
            item = self.id2item[i + 1]
            item2sem_ids[item] = (*tuple(sem_ids[i].tolist()), len(sem_id2item[str_id]))
            max_conflict = max(max_conflict, len(sem_id2item[str_id]))
        self.log(f'[TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        if max_conflict > self.codebook_sizes[-1]:
            raise ValueError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
                f'{max_conflict} > {self.codebook_sizes[-1]}. Please increase the codebook size.'
            )
        
        assert self.dedup_on_creation or self.dedup_on_load

        return item2sem_ids

    def _generate_semantic_rqvae(
        self,
        rqvae_model: RQVAEModel,
        sent_embs: torch.Tensor,
        sem_ids_path: str
    ) -> None:
        """
        Generates semantic IDs using the given RQVAE model and saves them to a file.

        Args:
            rqvae_model (RQVAEModel): The RQVAE model used for encoding sentence embeddings.
            sent_embs (torch.Tensor): The sentence embeddings to be encoded.
            sem_ids_path (str): The path to save the generated semantic IDs.

        Returns:
            None
        """
        rqvae_model.eval()
        rqvae_sem_ids = rqvae_model.encode(sent_embs)
        item2sem_ids = self._extend_semantic_ids(rqvae_sem_ids)
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_faiss(
        self,
        sent_embs: np.ndarray,
        sem_ids_path: str,
        train_mask: np.ndarray
    ) -> None:
        """
        Generates semantic IDs using the Faiss library and saves them to a file.

        Args:
            sent_embs (np.ndarray): The sentence embeddings.
            sem_ids_path (str): The path to save the semantic IDs.
            train_mask (np.ndarray): A boolean mask indicating which items are used for training.

        Returns:
            None
        """
        n_bits = int(np.log2(self.config['rq_codebook_size']))

        import faiss
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.IndexResidualQuantizer(
            sent_embs.shape[-1],
            self.config['rq_n_codebooks'],
            n_bits,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log('[TOKENIZER] Training index...')
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        faiss_sem_ids = []
        uint8_code = index.rq.compute_codes(sent_embs)
        n_bytes = uint8_code.shape[1]
        self.logger.info('[TOKENIZER] Generating semantic IDs...')
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            code = []
            for i in range(self.config['rq_n_codebooks']):
                code.append(bs.read(n_bits))
            faiss_sem_ids.append(code)
        faiss_sem_ids = np.array(faiss_sem_ids)
        item2sem_ids = self._extend_semantic_ids(faiss_sem_ids)
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_ids(self, sem_ids_path: str, dataset: AbstractDataset):
        # Load or encode sentence embeddings
        sent_emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )
        if os.path.exists(sent_emb_path):
            self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
        else:
            self.log('[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
        # PCA
        if self.config['sent_emb_pca'] > 0:
            self.log('[TOKENIZER] Applying PCA to sentence embeddings...')
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            sent_embs = pca.fit_transform(sent_embs)
        self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

        # Generate semantic IDs
        training_item_mask = self._get_items_for_training(dataset)
        if self.config['rq_faiss']:
            self.log('[TOKENIZER] Semantic IDs not found. Training index using Faiss...')
            self._generate_semantic_id_faiss(sent_embs, sem_ids_path, training_item_mask)
        else:
            self.log('[TOKENIZER] Semantic IDs not found. Training RQ-VAE model...')
            embs_for_training = torch.FloatTensor(sent_embs[training_item_mask]).to(self.config['device'])
            sent_embs = torch.FloatTensor(sent_embs).to(self.config['device'])
            model_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'rqvae_{list_to_str(self.codebook_sizes, remove_blank=True)}.pth'
            )
            rqvae_model = self._train_rqvae(embs_for_training, model_path)
            self._generate_semantic_rqvae(rqvae_model, sent_embs, sem_ids_path)
