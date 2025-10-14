from finetuning.utils import train_n_epochs
from sparsification.qpm_sparsification import compute_qpm_feature_selection_and_assignment


def finetune_chiqpm(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, n_features, n_per_class, lambda_feat= 0):
    feature_sel, weight,  mean, std = compute_qpm_feature_selection_and_assignment(model, train_loader,
                                                                                         test_loader,
                                                                                         log_dir, n_classes, seed, n_features, n_per_class, rho=0.5)
    model.set_model_sldd(feature_sel, weight, mean, std, retrain_normalisation=False, relu=True)
    for iteration_epoch in range(4): # To make use of Q-SENN LR Schedule
        model = train_n_epochs(model, beta, optimization_schedule, train_loader, test_loader, lambda_feat)
    return model