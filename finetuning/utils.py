from FeatureDiversityLoss import FeatureDiversityLoss
from train import train, test
from training.optim import get_optimizer


def train_n_epochs(model, beta,optimization_schedule, train_loader, test_loader, lambda_feat=0):
    optimizer, schedule, epochs = get_optimizer(model, optimization_schedule)
    fdl = FeatureDiversityLoss(beta, model.linear)
    end_lambda = None
    if isinstance(lambda_feat, tuple):
        lambda_feat, end_lambda = lambda_feat
    for epoch in range(epochs):
        if end_lambda is not None:
            if epoch == end_lambda:
                print("Turning feature grounding loss off at epoch ", epoch)
                lambda_feat = 0
        model = train(model, train_loader, optimizer, fdl, lambda_feat,epoch)
        schedule.step()
        if epoch % 5 == 0 or epoch+1 == epochs:
            test(model, test_loader, epoch)
    return model