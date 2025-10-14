import torch
from torchcp.classification.predictors import SplitPredictor

from conformalPrediction.cleanScoreFunction import HieraDiffNonConformityScore



class CustomHierarchicalConformityScorePredictor(SplitPredictor):
    def __init__(self, weight):
        # Finetune NoRandom Alph0.925_DDiffAdaX_10_Acc
        score = HieraDiffNonConformityScore(weight)
        super().__init__(score)

    def calculate_threshold(self, logits, labels, alpha, features):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        features = features.to(self._device)
        self.q_hat = 0

        alpha, mask = self.score_function.calibrate_alpha(logits, labels, alpha, features)
        logits = logits[mask]
        labels = labels[mask]
        features = features[mask]
        scores = self.score_function(logits, features, labels, )
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def predict_with_logits(self, logits, features, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """

        scores = self.score_function(logits, features)
        scores = scores.to(self._device)
        if q_hat is None:
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)
        return S