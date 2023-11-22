import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def from_logit_to_pred(out: torch.Tensor) -> torch.Tensor:
    if (len(out.shape) == 2) and (out.shape[1] >= 2):
        return out.argmax(dim=1).data
    elif len(out.shape) == 1:
        return (out >= 0).long()
    return None


def _f1_score_torch(average: str = "binary") -> float:
    """
    Calculated an average of ['micro', 'macro', 'samples', 'weighted', 'binary']
    """

    def f1_score_torch_(out: torch.Tensor, y: torch.Tensor):
        y_pred = from_logit_to_pred(out).numpy()
        y_true = y.numpy()
        return f1_score(y_true=y_true, y_pred=y_pred, average=average)

    return f1_score_torch_


def f1(out: torch.Tensor, y: torch.Tensor) -> float:
    return _f1_score_torch(average="binary")(out, y)


def f1_binary(out: torch.Tensor, y: torch.Tensor) -> float:
    return _f1_score_torch(average="binary")(out, y)


def f1_micro(out: torch.Tensor, y: torch.Tensor) -> float:
    return _f1_score_torch(average="micro")(out, y)


def f1_macro(out: torch.Tensor, y: torch.Tensor) -> float:
    return _f1_score_torch(average="macro")(out, y)


def f1_weighted(out: torch.Tensor, y: torch.Tensor) -> float:
    return _f1_score_torch(average="weighted")(out, y)


def accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
    """
    Accuracy classification score.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

    Args:
        out (torch.Tensor): Predicted labels, as returned by a model
        y (torch.Tensor): True labels.

    Returns:
        float: The fraction of correctly classified samples
    """
    y_pred = from_logit_to_pred(out).numpy()
    y_true = y.numpy()
    return accuracy_score(y_true, y_pred)


def roc_auc(out: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    Args:
        out (torch.Tensor): Predicted labels, as returned by a model
        y (torch.Tensor): True labels.

    Returns:
        float: Area Under the Curve score.
    """
    if (len(out.shape) == 2) and (out.shape[1] == 2):
        out = torch.nn.functional.softmax(out, dim=1)[:, 1]
    y_pred = out.data.numpy()
    y_true = y.numpy()
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        pass


def average_precision(out: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute average precision (AP) from prediction scores.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

    Args:
        out (torch.Tensor): Predicted labels, as returned by a model
        y (torch.Tensor): True labels.

    Returns:
        float: Average precision score.
    """
    if (len(out.shape) == 2) and (out.shape[1] == 2):
        out = torch.nn.functional.softmax(out, dim=1)[:, 1]
    y_pred = out.data.numpy()
    y_true = y.numpy()
    return average_precision_score(y_true, y_pred)


def get_metric(name: str) -> float:
    """
    name one of:
    'f1',
    'f1_binary',
    'f1_micro',
    'f1_macro',
    'f1_weighted',
    'accuracy',
    'roc_auc',
    'average_precision'
    and functions from torch.nn.functional
    """
    func = globals().get(name)
    if func is not None:
        return func
    elif hasattr(torch.nn.functional, name):
        return getattr(torch.nn.functional, name)
    else:
        raise NotImplementedError("no metric: " + name)
