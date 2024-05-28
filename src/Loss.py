from pytorch_metric_learning.distances import CosineSimilarity
import torch


class InfoNCELoss():
    def __init__(self, device, k, temperature=0.07, threshold=1.0, fna=False):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.similarity = CosineSimilarity()
        self.k = k
        self.temperature = temperature
        self.threshold = threshold
        self.fna = fna

    def __call__(self, x, y, labels, sts):
        false_negatives = self.detect_false_negative(sts)
        indices_tuple = self.get_all_pairs_indices(labels, false_negatives)

        mat = self.similarity(x, y)
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]

        if len(neg_pair) > 0 and self.k > -1:
            paired = list(zip(neg_pair.tolist(), a2.tolist(), n.tolist()))
            selected = sorted(paired, key=lambda x: x[0], reverse=True)[:self.k]
            _, x, y = list(zip(*selected))
            x = torch.tensor(x).to(a2.device)
            y = torch.tensor(y).to(n.device)

            neg_pair = mat[x, y]
            indices_tuple = (a1, p, x, y)

        return self._compute_loss(pos_pair, neg_pair, indices_tuple), len(pos_pair)

    def detect_false_negative(self, embs):
        mat = torch.matmul(embs, torch.t(embs))
        return torch.where(mat >= self.threshold)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype

            if not self.similarity.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = a2.unsqueeze(0) == a1.unsqueeze(1)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = torch.finfo(dtype).min

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + torch.finfo(dtype).tiny)
            return torch.mean(-log_exp)

        return 0

    def get_all_pairs_indices(self, labels, false_negatives):
        labels1 = labels.unsqueeze(1)
        labels2 = labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1

        diffs[false_negatives[0], false_negatives[1]] = 0  # FNE
        if self.fna:
            matches[false_negatives[0], false_negatives[1]] = 1  # FNA

        diffs.fill_diagonal_(0)
        matches.fill_diagonal_(1)

        a1_idx, p_idx = torch.where(matches)
        a2_idx, n_idx = torch.where(diffs)
        return a1_idx, p_idx, a2_idx, n_idx
