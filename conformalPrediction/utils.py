import torch
from scipy.signal import sweep_poly
from torchcp import classification
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import THR, APS

from conformalPrediction.CustomPredictor import CustomHierarchicalConformityScorePredictor


def calibrate_predictor(logits, labels, confidence, features, predictor, needs_feats):
    if needs_feats:
        predictor.calculate_threshold(logits, labels, 1 - confidence, features)
    else:
        predictor.calculate_threshold(logits, labels, 1 - confidence, )


def get_predictions(logits, predictor, features, need_feats):
    if need_feats:
        return predictor.predict_with_logits(logits, features)
    else:
        return predictor.predict_with_logits(logits)


def get_score(score, weight):
    func = None
    if score == "THR":
        func = THR()
    elif score == "APS":
        func = APS()
    if func is not None:
        return SplitPredictor(score_function=func), False
    return CustomHierarchicalConformityScorePredictor(weight), True


