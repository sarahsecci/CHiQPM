from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.hipify.hipify_python import mapping

from configs.dataset_params import normalize_params
from dataset_classes.cub200 import CUB200Class, load_cub_class_mapping
from evaluation.load_model import load_model, get_args_for_loading_model
from get_data import get_data
from visualization.get_heatmaps import get_visualizations
from configs.dataset_params import normalize_params
from visualization.pairstoViz import find_easier_interpretable_pairs, select_clearly_activating_separable_samples
from visualization.utils import get_active_mean


def get_combined_indices(combined_indices):
    # Returns feature indices joined, so that first uniques of class 1, then shared, then uniques of class two.

    if len(combined_indices) > 2:
        rel_values = list(combined_indices.values())
        total_indices = set()
        for rel_value in rel_values:
            total_indices = total_indices.union(rel_value)
        total_indices = sorted(list(total_indices))
    else:

        rel_values = list(combined_indices.values())
        shared_indices = set(rel_values[0]).intersection(rel_values[1])
        middle_indices = sorted(list(shared_indices))
        unique_first = [i for i in rel_values[0] if i not in shared_indices]
        unique_second = [i for i in rel_values[1] if i not in shared_indices]
        total_indices = unique_first + middle_indices + unique_second
    return total_indices

def viz_model(model,data_loader, class_indices ,test_key, active_mean=None,gamma = 3, norm_across_channels = False, size=(2.5,2.5), viz_folder = None):
    images = []
    image_unnormalized = []
    data_mean, data_std = normalize_params[data_loader.dataset.name]["mean"], normalize_params[data_loader.dataset.name]["std"]
    assert len(class_indices) >= 2
    class_names = None
    if isinstance(data_loader.dataset, CUB200Class):
        mapping = load_cub_class_mapping()
        class_names = [mapping[str(x)] for x in class_indices]
    combined_indices = {}
    for j, c_index in enumerate(class_indices):
        rel_indices = data_loader.dataset.get_indices_for_target(c_index)
        class_features = model.linear.weight[c_index].nonzero().flatten().tolist()
        combined_indices[c_index] = class_features
        class_images = []
        for idx in rel_indices:
            image, label = data_loader.dataset[idx]
            assert label == c_index
            class_images.append(image)
        image = select_clearly_activating_separable_samples(model, class_images, c_index)
        image_unnormalized.append(image* data_std[:, None, None] + data_mean[:, None, None])

        images.append(image)
    combined_indices = get_combined_indices(combined_indices)
    img_full = torch.stack(images)
    img_full = img_full.to("cuda")
    image_unnormalized = torch.stack(image_unnormalized)
    visualizations, colormapping = get_visualizations(combined_indices, img_full,image_unnormalized, model, gamma=gamma,
                                               norm_across_images=norm_across_channels,active_means=active_mean, with_color=True )
    fig, axes  = plt.subplots(len(class_indices), len(visualizations) + 1, figsize=(size[0]*(len(visualizations) +1), size[1] * len(class_indices)))

    for i, img in enumerate(image_unnormalized):
        ax = axes[i,0]
        ax.imshow(img.permute(1, 2, 0))
        if class_names is not None:
            ax.set_ylabel(class_names[i])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    for i, feat_maps_for_samples in enumerate(visualizations):
        for j, feat_for_sample_viz in enumerate(feat_maps_for_samples):
            ax = axes[j, i+1]
            ax.imshow(feat_for_sample_viz.permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    if viz_folder is None:
        viz_folder = Path.home() / "tmp" / f"{test_key}vizQPMClassComparisons"
    viz_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(viz_folder / f"{'_'.join([str(x) for x in class_indices])}.png",  bbox_inches='tight')
    return colormapping



if  __name__ == "__main__":
    args = get_args_for_loading_model()
    train_loader, test_loader= get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    model, folder  = load_model(args.dataset, args.arch, args.seed, args.model_type, args.cropGT, args.n_features,
                       args.n_per_class, args.img_size, args.reduced_strides, args.folder)
    interesting_pairs = find_easier_interpretable_pairs(model, train_loader,min_sim= 0.8)
    if args.dataset == "CUB2011":
        mapping = load_cub_class_mapping()
        class_names = [(mapping[str(x)], mapping[str(y)]) for x, y in interesting_pairs]
    train_loader.dataset.transform = test_loader.dataset.transform
    active_mean = get_active_mean(model, train_loader, folder   )
    for pair in interesting_pairs:
        class_indices = list(pair)
        viz_model(model, test_loader, class_indices, "Test", active_mean)

        viz_model(model, train_loader, class_indices, "Train", active_mean)