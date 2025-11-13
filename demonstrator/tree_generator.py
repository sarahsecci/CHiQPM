"""
Tree generation module for CHiQPM demo.
Handles conformal prediction setup, calibration, and tree visualization.
"""

import types
import torch
from PIL import Image
from pathlib import Path

from conformalPrediction.visualize_tree import HierarchicalExplainer
from conformalPrediction.utils import get_score, calibrate_predictor, get_predictions


class TreeGenerator:
    """
    Handles all tree generation logic including conformal prediction setup.
    Returns PIL images instead of saving to disk for better performance.
    """
    
    def __init__(self, model, calibration_data, class_names):
        """
        Initialize tree generator.
        
        Args:
            model: The trained CHiQPM model
            calibration_data: Dict with 'cal_logits', 'cal_labels', 'cal_features'
            class_names: Dict mapping class indices to names
        """
        self.model = model
        self.calibration_data = calibration_data
        self.class_names = class_names
        self.explainer = HierarchicalExplainer(model.linear.weight)
        
    def _apply_conformal_fix(self, predictor):
        """
        Apply monkey-patch fix to the predictor's nonconformity score function.
        This fixes the limit_delta calculation for proper conformal prediction.
        """
        def fixed_nonconformity_score(self, logits, features):
            logits = logits.to(self.weight.device)
            features = features.to(self.weight.device)
            delta_n, sorted_values = self.get_delta_n_up(logits, features)
            contains_one_mismatch = (delta_n.sum(dim=1) < self.weights_per_class)
            mismatch_position = torch.argmin(delta_n.float(), dim=1)

            masked_for_upscore = sorted_values * delta_n
            B, K, C = sorted_values.shape
            device = sorted_values.device
            batch_idx = torch.arange(B, device=device).view(B, 1)
            class_idx = torch.arange(C, device=device).view(1, C)

            value_at_diversion = sorted_values[batch_idx, mismatch_position, class_idx] * contains_one_mismatch

            limit_delta = torch.ones(B, C, device=device)
            if self.lvl > 0:
                limit_delta = delta_n[:, self.lvl - 1]
            
            limited_up_score = -masked_for_upscore[:, self.lvl:].sum(dim=1) - value_at_diversion
            limited_up_score_masked = limited_up_score * limit_delta
            return limited_up_score_masked

        predictor.score_function.nonconformity_score_for_every_class = types.MethodType(
            fixed_nonconformity_score, 
            predictor.score_function
        )
        
    def _setup_predictor(self, accuracy):
        """
        Create and calibrate a conformal predictor for the given accuracy level.
        
        Args:
            accuracy: Target accuracy level (e.g., 0.9 for 90%)
            
        Returns:
            Calibrated predictor and prediction set for the sample
        """
        predictor, needs_feats = get_score("CHiQPM", self.model.linear.weight)
        self._apply_conformal_fix(predictor)
        
        calibrate_predictor(
            self.calibration_data['cal_logits'],
            self.calibration_data['cal_labels'],
            accuracy,
            self.calibration_data['cal_features'],
            predictor,
            needs_feats
        )
        
        return predictor, needs_feats
    
    def generate_tree(self, output_logits, final_features, accuracy, colormapping):
        """
        Generate tree visualization as PIL Image (no file saving).
        
        Args:
            output_logits: Model output logits [1, n_classes]
            final_features: Final layer features [1, n_features]
            accuracy: Target accuracy level (0.8-0.999)
            colormapping: Feature to color mapping dict
            
        Returns:
            PIL.Image of the tree visualization, or None if generation fails
        """
        predictor, needs_feats = self._setup_predictor(accuracy)
        
        prediction_set = get_predictions(
            output_logits,
            predictor,
            final_features,
            needs_feats
        )[0]
        
        # Try local tree first, fallback to global if KeyError
        try:
            tree_image = self.explainer.generate_explanation_to_pil(
                output_logits.squeeze(0),
                final_features.squeeze(0),
                prediction_set,
                gt_label=None,
                feature_to_color_mapping=colormapping,
                global_plot=False,
                class_names=self.class_names
            )
            return tree_image
            
        except KeyError as e:
            print(f"Local tree failed for accuracy {accuracy:.3f}, using global tree instead")
            try:
                tree_image = self.explainer.generate_explanation_to_pil(
                    output_logits.squeeze(0),
                    final_features.squeeze(0),
                    prediction_set,
                    gt_label=None,
                    feature_to_color_mapping=colormapping,
                    global_plot=True,
                    class_names=self.class_names
                )
                return tree_image
            except Exception as e2:
                print(f"Tree generation failed: {e2}")
                return None
