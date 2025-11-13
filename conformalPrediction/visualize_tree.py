from pathlib import Path

import torch

from conformalPrediction.HierarchicalExplanation.graphCode import visualize_explanation_tree, visualize_explanation_tree_to_pil
from conformalPrediction.cleanScoreFunction import HieraDiffNonConformityScore
from conformalPrediction.eval_cp import get_logits_and_labels
from conformalPrediction.utils import get_score, calibrate_predictor, get_predictions
from dataset_classes.cub200 import load_cub_class_mapping, CUB200Class
from evaluation.load_model import get_args_for_loading_model, load_model
from get_data import get_data
from visualization.compare_classes import viz_model
from visualization.localViz import generate_local_viz
from visualization.utils import get_feats_logits_labels, get_active_mean


class HierarchicalExplainer(HieraDiffNonConformityScore):
    def __init__(self, weight):
        super().__init__(weight)
        self.n_classes = weight.shape[0]
        self.class_class_similarity = (weight @ weight.T)

    def get_most_similar_classes(self, pred, max_classes = 5):
        class_sim = self.class_class_similarity[pred]
        sorted_similarities, sorted_indices = torch.sort(class_sim, descending=True)
        max_sim = sorted_similarities[1]
        answer = [pred, sorted_indices[1].item()]
        if max_sim == self.weights_per_class -1:
            equal_to_max = (sorted_similarities == max_sim).nonzero().flatten()
            for idx in equal_to_max[1:]:

                answer.append(sorted_indices[idx].item())
                if len(answer) >= max_classes:
                    break
        return tuple(answer)

    def generate_explanation(self, logits,features,prediction_set, gt_label = None, feature_to_color_mapping=None, folder = None, filename = None, global_plot = False, class_names = None):
        features = features.to(self.weight.device)
        sorted_values, sorted_indices = self.get_sorted_indices_and_values(features[None])
        pred = torch.argmax(logits)[None]
        delta_n = self.get_shared_with_preds_from_sorted(sorted_indices, pred)
        values_for_this_sample = sorted_values[0]
        indices_for_this_sample = sorted_indices[0]
        delta_n_for_this_sample = delta_n[0]

        hierarchy_data = [] # This is a list with one dict per level in the hierarchy. Level refers to numbers of features
        global_hierarchy_data = []

        # Local hierarchy data only includes classes that share the most important feature with the predicted class (delta_n[0] is True])
        for level in range(self.weights_per_class):
            level_dict = {}
            global_level_dict = {}
            for class_idx in range(self.n_classes):
                delta_n_for_this_class = self.get_shared_with_preds_from_sorted(sorted_indices, class_idx)
                predicted_classes_down_the_road = set(delta_n_for_this_class[0, level].nonzero().flatten().tolist())
                this_f_idx = indices_for_this_sample[level, class_idx].item()
                act = values_for_this_sample[level, class_idx].item()
                sharing_at_first_level = delta_n_for_this_sample[0, class_idx].item()
                prev = None
                if level > 0:
                    prev = tuple(indices_for_this_sample[:level, class_idx].tolist())

                tuple_key =(this_f_idx, act,prev ,class_idx)
                global_level_dict[tuple_key] = predicted_classes_down_the_road
                if sharing_at_first_level:
                    level_dict[tuple_key] = global_level_dict[tuple_key]
            hierarchy_data.append(level_dict)
            global_hierarchy_data.append(global_level_dict)
        
        if class_names is None:
            class_names={}
            
        if global_plot:
            visualize_explanation_tree(global_hierarchy_data, gt_label, class_names, features, pred.item(),
                                       prediction_set, folder, filename, feature_to_color_mapping)
        else:

              visualize_explanation_tree(hierarchy_data, gt_label,class_names, features,pred.item(),prediction_set,  folder, filename, feature_to_color_mapping,global_plot = global_plot)

        pass

    def generate_explanation_to_pil(self, logits, features, prediction_set, gt_label=None, feature_to_color_mapping=None, global_plot=False, class_names=None):
        """
        Generate explanation tree and return as PIL Image (no file saving).
        
        This is a memory-efficient version for interactive applications like Gradio demos.
        
        Args:
            logits: Model output logits for the sample
            features: Feature activations for the sample
            prediction_set: Set of predicted class indices from conformal prediction
            gt_label: Ground truth label (optional)
            feature_to_color_mapping: Dict mapping feature indices to colors
            global_plot: If True, generate global tree; if False, generate local tree
            class_names: Dict mapping class indices to names
            
        Returns:
            PIL.Image: The rendered tree visualization
        """
        features = features.to(self.weight.device)
        sorted_values, sorted_indices = self.get_sorted_indices_and_values(features[None])
        pred = torch.argmax(logits)[None]
        delta_n = self.get_shared_with_preds_from_sorted(sorted_indices, pred)
        values_for_this_sample = sorted_values[0]
        indices_for_this_sample = sorted_indices[0]
        delta_n_for_this_sample = delta_n[0]

        hierarchy_data = []
        global_hierarchy_data = []

        # Build hierarchy data
        for level in range(self.weights_per_class):
            level_dict = {}
            global_level_dict = {}
            for class_idx in range(self.n_classes):
                delta_n_for_this_class = self.get_shared_with_preds_from_sorted(sorted_indices, class_idx)
                predicted_classes_down_the_road = set(delta_n_for_this_class[0, level].nonzero().flatten().tolist())
                this_f_idx = indices_for_this_sample[level, class_idx].item()
                act = values_for_this_sample[level, class_idx].item()
                sharing_at_first_level = delta_n_for_this_sample[0, class_idx].item()
                prev = None
                if level > 0:
                    prev = tuple(indices_for_this_sample[:level, class_idx].tolist())

                tuple_key = (this_f_idx, act, prev, class_idx)
                global_level_dict[tuple_key] = predicted_classes_down_the_road
                if sharing_at_first_level:
                    level_dict[tuple_key] = global_level_dict[tuple_key]
            hierarchy_data.append(level_dict)
            global_hierarchy_data.append(global_level_dict)

        if class_names is None:
            class_names = {}

        # Use the new PIL-returning function
        if global_plot:
            return visualize_explanation_tree_to_pil(
                global_hierarchy_data, gt_label, class_names, features, pred.item(),
                prediction_set, feature_to_color_mapping, global_plot=True
            )
        else:
            return visualize_explanation_tree_to_pil(
                hierarchy_data, gt_label, class_names, features, pred.item(),
                prediction_set, feature_to_color_mapping, global_plot=False
            )





if __name__ == '__main__':
    args = get_args_for_loading_model()
    train_loader, test_loader = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    train_loader.dataset.transform = test_loader.dataset.transform
    model, folder = load_model(args.dataset, args.arch, args.seed, args.model_type, args.cropGT, args.n_features,
                               args.n_per_class, args.img_size, args.reduced_strides, args.folder)
    features_test, outputs_test, labels_test = get_feats_logits_labels(model, test_loader)
    cal_logits, cal_labels, cal_features, test_logits, test_labels, test_features, test_indices = get_logits_and_labels(
        features_test, outputs_test, labels_test, 10, )
    graph_folder = Path.home() / "tmp" / "CHiQPMExplanations"
    # Since the examples chosen in readme are for classes with indices 25,26, 176 and 181 we only show these
    test_labels_in_readme = [25, 26, 176, 181]
    kept_label = torch.zeros_like(test_labels)
    for lab in test_labels_in_readme:
        kept_label = kept_label + (test_labels == lab).long()
    test_logits = test_logits[kept_label.bool()]
    test_features = test_features[kept_label.bool()]
    test_labels = test_labels[kept_label.bool()]
    index_in_dataset = test_indices[kept_label.bool()]
    if args.dataset == "CUB2011":
        class_names = load_cub_class_mapping()
        class_names = {int(k): v for k, v in class_names.items()}
    # Reasonable cherrypicks for this model
    indices_of_test_samples = [ 23,43,3,59]

    explainer = HierarchicalExplainer(model.linear.weight)
    for acc in [ .9,.95,  .925,.88, ]: #


        predictor, needs_feats = get_score("CHiQPM", model.linear.weight)

        calibrate_predictor(cal_logits, cal_labels, acc, cal_features, predictor, needs_feats)


        prediction_sets = get_predictions(test_logits, predictor, test_features, needs_feats)
        if indices_of_test_samples != []:
            for sample in indices_of_test_samples:


                prediction = torch.argmax(test_logits[sample]).item()
                class_pair_interesting = explainer.get_most_similar_classes(prediction,)
                index_in_dataset_of_sample = index_in_dataset[sample].item()
                filename_end = f"Sample_{index_in_dataset_of_sample}_label_{class_names[test_labels[sample].item()]}"
                image = test_loader.dataset.__getitem__(index_in_dataset_of_sample)[0]
                graph_folder_acc = graph_folder / f"SampleIndex_{str(index_in_dataset_of_sample)}" / f"acc_{acc}"
                graph_folder_acc.mkdir(parents=True, exist_ok=True)
                image_unnormalized = (image * test_loader.dataset.transform.transforms[-1].std[:, None, None] +
                                      test_loader.dataset.transform.transforms[-1].mean[:, None, None])
                active_mean = get_active_mean(model, train_loader, folder)
                class_indices = list(class_pair_interesting)
                colormapping = viz_model(model, train_loader, class_indices, "Train",
                                         active_mean, viz_folder=graph_folder_acc)
                generate_local_viz(image, image_unnormalized, model, colormapping,graph_folder_acc /
                                   f"Local_{filename_end}.png",active_mean=active_mean )

                explainer.generate_explanation(test_logits[sample], test_features[sample], prediction_sets[sample],feature_to_color_mapping=colormapping, gt_label = test_labels[sample], folder = graph_folder_acc, filename = f"Graph_{filename_end}")
        else:
            graph_folder_ac = graph_folder / f"acc_{acc}"
            graph_folder_ac.mkdir(parents=True, exist_ok=True)
            for sample in range(len(test_logits)):
                explainer.generate_explanation(test_logits[sample], test_features[sample], prediction_sets[sample], gt_label = test_labels[sample], folder = graph_folder_ac, filename = f"sample_{sample}_label_{test_labels[sample].item()}")

