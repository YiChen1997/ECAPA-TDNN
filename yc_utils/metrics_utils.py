from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)


def get_train_val_metrics(y_true, y_pred, prefix=None):
    """
    Return a dictionary of classification metrics
    """
    init_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }
    if prefix is not None:
        init_metrics = {f"{prefix}/{k}": v for k, v in init_metrics.items()}
    return init_metrics


def get_test_metrics(
        scores, labels, mindcf_p_target=1e-2, mindcf_c_fa=1, mindcf_c_miss=1, prefix=None
):
    """
    Return EER and minDCF metrics
    """
    init_metrics = {
        "eer": compute_eer(scores, labels),
        "mindcf": compute_mindcf(
            scores,
            labels,
            p_target=mindcf_p_target,
            c_fa=mindcf_c_fa,
            c_miss=mindcf_c_miss,
        ),
    }
    if prefix is not None:
        init_metrics = {f"{prefix}/{k}": v for k, v in init_metrics.items()}
    return init_metrics


# TODO 比较这里的compute_eer和 minDCF
def compute_eer(scores, labels):
    """
    Compute the equal error rate score
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def compute_error_rates(scores, labels, eps=1e-6):
    """
    Creates a list of false negative rates, a list of false positive rates
    and a list of decision thresholds that give those error rates
    (see https://github.com/clovaai/voxceleb_trainer)
    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the error-rates are evaluated.
    sorted_indexes, _ = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=lambda t: t[1],
        )
    )
    labels = [labels[i] for i in sorted_indexes]

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i] and fprs[i]
    # is the total number of times that we have correctly accepted
    # scores greater than thresholds[i]
    fnrs, fprs = [], []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / (float(fnrs_norm) + eps) for x in fnrs]

    # Divide by the total number of correct positives to get the
    # true positive rate and subtract these quantities from 1 to
    # get the false positive rates
    fprs = [1 - x / (float(fprs_norm) + eps) for x in fprs]

    return fnrs, fprs


def compute_mindcf(scores, labels, p_target=1e-2, c_fa=1, c_miss=1, eps=1e-6):
    """
    Computes the minimum of the detection cost function
    (see https://github.com/clovaai/voxceleb_trainer)
    """
    # Extract false negative and false positive rates
    fnrs, fprs = compute_error_rates(scores, labels)

    # Compute the minimum detection cost
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    # Compute default cost and use it to normalize the
    # minimum detection cost
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / (c_def + eps)

    return min_dcf

