import numpy as np
import torch
from torchcp.classification.scores.base import BaseScore




class HieraDiffNonConformityScore(BaseScore):

    def __init__(self, weight):
        self.weight = weight
        self.weights_per_class = int(weight.sum(dim=1).mean().item())
        super().__init__()

    def _find_highest_sufficient_level(self, sufficient_acc):
        """
        Find the highest hierarchical level where accuracy is sufficient.
        
        Args:
            sufficient_acc: Boolean tensor indicating which levels satisfy the requirement
            
        Returns:
            int: The highest level index where sufficient_acc is True
        """
        if sufficient_acc.any():
            if sufficient_acc.all():
                return len(sufficient_acc) - 1
            else:
                return torch.argmin(sufficient_acc.float()).item()
        else:
            print("Warning: No level satisfies the accuracy constraint")
            return 0

    def compute_scores_for_all_levels(self, logits, labels, features):
        """
        Pre-compute nonconformity scores for all hierarchical levels.
        
        Args:
            logits: calibration logits
            labels: calibration labels  
            features: calibration features
            
        Returns:
            dict: {level: scores_tensor}
        """
        original_level = getattr(self, 'level', 0)
        
        scores_per_level = {}
        for level in range(self.weights_per_class):
            self.level = level
            scores = self(logits, features, labels)
            scores_per_level[level] = scores
        
        self.level = original_level
        
        return scores_per_level

    def get_adaptive_level(self, logits, labels, alpha, features):
        delta_n,_= self.get_delta_n_up(logits, features)
        correct_label_included = delta_n[torch.arange(len(labels)), :,labels]
        total_corrects = correct_label_included.sum(dim=0)
        self.total_accs = total_corrects / len(labels)
        sufficient_acc = self.total_accs >= (1 - alpha)
        return self._find_highest_sufficient_level(sufficient_acc)
    

    def calibrate_alpha(self, logits, labels, alpha, features):

        self.level = self.get_adaptive_level(logits, labels, alpha, features)
        mask = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)

        return alpha, mask


    def _calculate_all_label(self, logits, features):
        return self.nonconformity_score_for_every_class(logits, features)

    def get_sorted_indices_and_values(self, features):
        # eq. 7
        masked_features = features[..., None] * self.weight.T + torch.ones_like(features[..., None]) * -1 * (
                1 - self.weight.T)
        sorted_values, sorted_indices = torch.sort(masked_features, dim=1, descending=True)
        sorted_values, sorted_indices = sorted_values[:, :self.weights_per_class], sorted_indices[
            :, :self.weights_per_class]
        return sorted_values, sorted_indices

    def get_shared_with_preds_from_sorted(self,  sorted_indices, actual_preds):


        # eq. 8
        if isinstance(actual_preds,int): # during Explanation generation
            label_indices = sorted_indices[:, :, actual_preds]

        else:

            label_indices = sorted_indices[torch.arange(len(actual_preds)), :, actual_preds]

        same_as_pred_feat = label_indices[..., None] == sorted_indices
        delta_n = torch.cumprod(same_as_pred_feat, dim=1).bool()
        return delta_n

    def get_delta_n_up(self, logits, features):
        sorted_values, sorted_indices = self.get_sorted_indices_and_values(features)
        actual_preds = torch.argmax(logits, dim=1)
        delta_n = self.get_shared_with_preds_from_sorted( sorted_indices, actual_preds)
        return delta_n,sorted_values
    def nonconformity_score_for_every_class(self, logits, features, ):
        logits = logits.to(self.weight.device)
        features = features.to(self.weight.device)
        delta_n,  sorted_values = self.get_delta_n_up(logits, features)
        contains_one_mismatch = (delta_n.sum(dim=1) < self.weights_per_class)
        mismatch_position = torch.argmin(delta_n.float(), dim=1)


        masked_for_upscore = sorted_values * delta_n
        # eq. 10 would be sum over above
        B, K, C = sorted_values.shape
        device = sorted_values.device
        batch_idx = torch.arange(B, device=device).view(B, 1)  # Shape: (B, 1)
        class_idx = torch.arange(C, device=device).view(1, C)  # Shape: (1, C)

        # eq. 11 involves adding point of diversion. 0 if no mismatch
        value_at_diversion = sorted_values[batch_idx, mismatch_position, class_idx] * contains_one_mismatch

        # include eq. 12, limiting maximum level
        limit_delta = torch.ones(B, C, device=device)
        if self.level > 0:
            limit_delta = delta_n[:,self.level - 1]  # 1 indicates we only predict with one shared, hence requiring sharing at index 0
        limited_up_score = -masked_for_upscore[:, self.level:].sum(dim=1) - value_at_diversion # eq.11
        limited_up_score_masked = limited_up_score * limit_delta
        return limited_up_score_masked

    def _calculate_single_label(self, logits, features, labels):
        limited_up_score_masked = self.nonconformity_score_for_every_class(logits, features)
        nonconformity_scores_of_target = limited_up_score_masked[torch.arange(len(labels)), labels]
        return nonconformity_scores_of_target


    def __call__(self, logits, features, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs, features)
        else:
            return self._calculate_single_label(probs, features, label)