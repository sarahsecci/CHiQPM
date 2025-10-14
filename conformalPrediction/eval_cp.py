import numpy as np
import torch


def get_logits_and_labels( features, logits, labels, cal_size,  ):
    device = features.device
    samples_per_class = np.ceil(len(features) // logits.shape[1])
    cal_indices = [i for i in range(len(features)) if (i % samples_per_class) < cal_size]
    keep_indices = [i for i in range(len(features)) if (i % samples_per_class) >= cal_size]
    with torch.no_grad():
        cal_logits = torch.tensor(logits[cal_indices], device=device)
        cal_labels = torch.tensor(labels[cal_indices], device=device)
        cal_features = torch.tensor(features[cal_indices], device=device)
        test_logits = torch.tensor(logits[keep_indices], device=device)
        test_labels = torch.tensor(labels[keep_indices], device=device)
        test_features = torch.tensor(features[keep_indices], device=device)
        # This has no impact for CHiQPM, as we use Relu, but it is fairly important for QPM to have a reasonable hierarchical baseline
        # Note that this does not impact other CP methods, as logits are unaffected
        cal_features = cal_features - cal_features.min()
        test_features = test_features - test_features.min()
        return cal_logits, cal_labels, cal_features, test_logits, test_labels, test_features, torch.tensor(keep_indices)
