import os

import torch

from evaluation.Metrics.Contrastiveness import gmm_metrics

def get_feats_logits_labels(model, loader):
    features = []
    logits = []
    labels = []
    model = model.to("cuda")
    model.eval()
    with torch.no_grad():
        for (data, target) in loader:
            xs1 = data.to("cuda")
            output, feature_maps, final_features = model(xs1, with_feature_maps=True, with_final_features=True, )
            features.append(final_features.to("cpu"))
            logits.append(output.to("cpu"))
            labels.append(target.to("cpu"))
    features = torch.concatenate(features)
    logits = torch.concatenate(logits)
    labels = torch.concatenate(labels)
    return features, logits, labels


def get_active_mean(model, train_loader, folder):
    if os.path.exists(os.path.join(folder, 'active_mean.npy')):
        return  torch.from_numpy(  torch.load(os.path.join(folder, 'active_mean.npy')))
    features, logits, labels = get_feats_logits_labels(model, train_loader)
    overlap, means, variances, variance_of_means = gmm_metrics(features)
    torch.save(means, os.path.join(folder, 'active_mean.npy'))
    return means