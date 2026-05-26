import random
import torch


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k,
            'err': self.err_at_k
        }

        self.legal_preds = set()
        for iid, tokens in self.tokenizer.item2tokens.items():
            self.legal_preds.add(tokens)

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()    # [B, K, C]
        labels = labels.detach().cpu()  # [B, C'] - C' is C+1 in TIGER
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        _, K, C = preds.shape
        eos_mask = labels[:, :C] == self.eos_token
        preds[eos_mask.unsqueeze(1).expand(-1, K, -1)] = self.eos_token
        return (preds == labels[:, None, :C]).all(-1)

    def calculate_pos_index_with_collisions(self, preds, labels, collision_cnts):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        collision_cnts = collision_cnts.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        offsets = torch.nn.functional.pad(collision_cnts.cumsum(-1), pad=(1, 0))
        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                pos_offset = offsets[i, j]
                if pos_offset >= self.maxk:
                    break
                sampled_j = random.randrange(pos_offset, offsets[i, j+1])
                if sampled_j >= self.maxk:
                    break
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, sampled_j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def err_at_k(self, preds, k):
        """
        Calculate the percentage illegal predictions
        among the top k generated token sequences.
        """
        ret = []
        for i in range(preds.shape[0]):
            n_illegal_preds = 0
            for j in range(k):
                cur_pred = tuple(preds[i, j].tolist())
                if cur_pred not in self.legal_preds:
                    n_illegal_preds += 1
            ret.append(n_illegal_preds / k)
        return torch.FloatTensor(ret)

    def calculate_metrics(self, preds, labels):        
        results = {}
        if 'collision_cnts' in preds and preds['collision_cnts'].max() > 1:
            pos_index = self.calculate_pos_index_with_collisions(preds['preds'], labels, preds['collision_cnts'])
        else:
            pos_index = self.calculate_pos_index(preds['preds'], labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        return results
