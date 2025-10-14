from matplotlib import pyplot as plt

from visualization.get_heatmaps import get_visualizations


def generate_local_viz(img_full, image_unnormalized, model, feature_to_color_mapping,  filename,  active_mean=None,gamma = 3,  size=(2.5,2.5),):
    combined_indices = list(feature_to_color_mapping.keys())
    if len(img_full.shape) == 3:
        img_full = img_full[None].to("cuda")
        image_unnormalized = image_unnormalized[None]
    visualizations, colormapping = get_visualizations(combined_indices, img_full, image_unnormalized, model,
                                                      gamma=3,
                                                      norm_across_images=False, active_means=active_mean,
                                                      with_color=True, colormapping = feature_to_color_mapping)
    fig, axes = plt.subplots(1, len(visualizations) + 1, figsize=(size[0] * (len(visualizations) + 1), size[1] * 2))

    for i, img in enumerate(image_unnormalized):
        ax = axes[i]
        ax.imshow(img.permute(1, 2, 0))
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    for j, feat_for_sample_viz in enumerate(visualizations):
        ax = axes[j + 1]
        ax.imshow(feat_for_sample_viz[0].permute(1, 2, 0))
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    filename.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(filename, bbox_inches='tight')