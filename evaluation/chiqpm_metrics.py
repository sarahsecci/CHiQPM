import numpy as np
import sklearn
import torch
from torchcp.classification import Metrics
from tqdm import trange

from conformalPrediction.utils import get_predictions, get_score, calibrate_predictor
from conformalPrediction.eval_cp import get_logits_and_labels
from evaluation.Metrics.Contrastiveness import gmm_overlap_per_feature
from evaluation.Metrics.Correlation import get_correlation
from evaluation.Metrics.Dependence import compute_contribution_top_feature, compute_class_independence
from evaluation.Metrics.SetCoherence import get_set_coherence
from evaluation.Metrics.StructuralGrounding import get_structural_grounding_for_weight_matrix
from evaluation.diversity import MultiKCrossChannelMaxPooledSum
from evaluation.qpm_metrics import evaluateALLMetricsForComps
from evaluation.utils import get_metrics_for_model

def evaluate_ChiQPMMetrics(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train, labels_test, features_test):
    cp_set_metrics = get_set_metrics(features_test, outputs_test,labels_test, linear_matrix)
    print("CP Set Metrics: ", cp_set_metrics)
    qpm_answer_dict = evaluateALLMetricsForComps(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train, labels_test, features_test)
    qpm_answer_dict.update(cp_set_metrics)
    return qpm_answer_dict


def get_set_metrics(features_test, outputs_test,labels_test,  weight):
    answer = {}
    # TODO split features into calibration and test sets
    cal_logits, cal_labels, cal_features, test_logits, test_labels, test_features, test_indices = get_logits_and_labels(
        features_test, outputs_test, labels_test, 10,)
    for method in [ "CHiQPM", "APS", "THR",]:
        answer[method] = {}
        for acc in [.88, .9, .925, .95]:
            answer[method][acc] = {}

            predictor, needs_feats = get_score(method, weight)

            calibrate_predictor(cal_logits, cal_labels, acc, cal_features, predictor, needs_feats)

            pass
            prediction_sets = get_predictions(test_logits, predictor, test_features, needs_feats)
            metrics = Metrics()
            Coverage_rate = metrics("coverage_rate")(prediction_sets, test_labels)
            Average_size = metrics("average_size")(prediction_sets, test_labels)
            answer[method][acc]["Coverage_rate"] = Coverage_rate
            answer[method][acc]["Average_size"] = Average_size
            # TODO, calibrate on calibration set, return predictions for test set
            # TODO store in answer dict
            if len(features_test) == 5794:
                answer[method][acc]["SetCoherence"] = get_set_coherence(prediction_sets)
    return answer

def eval_model_on_all_chiqpm_metrics(model, test_loader, train_loader):
    # TODO evaluate model on all metrics
    # TODO return metrics
    return get_metrics_for_model(train_loader, test_loader, model, evaluate_ChiQPMMetrics)
