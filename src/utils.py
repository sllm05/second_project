import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    # calculate f1 score using sklearn's function
    f1 = f1_score(labels, preds, average="micro")

    return {
        "accuracy": acc,
        "f1": f1,
    }