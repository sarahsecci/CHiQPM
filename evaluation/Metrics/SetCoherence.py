import numpy as np
import torch

from evaluation.Metrics.StructuralGrounding import get_class_sim_gt


def get_set_coherence(prediction_sets):
    class_sims = get_class_sim_gt()
    all_similarities = []
    for pred_set in prediction_sets:
        if len(pred_set) < 2:
            continue
        these_sims = class_sims[pred_set][:, pred_set]
        # Remove diagonal
        these_sims = these_sims[~torch.eye(len(pred_set), dtype=bool)].reshape(-1)
        all_similarities.extend(these_sims)
    return np.mean(all_similarities)