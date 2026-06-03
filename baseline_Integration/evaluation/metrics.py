"""
评估指标计算（精确率、召回率、F1、准确率）
"""

def compute_precision(tp: int, fp: int) -> float:
    """精确率 = TP / (TP + FP)"""
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def compute_recall(tp: int, fn: int) -> float:
    """召回率 = TP / (TP + FN)"""
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def compute_f1(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R)"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """准确率 = (TP + TN) / 总样本数"""
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total

def compute_metrics(preds, labels):
    """
    根据预测结果和真实标签计算所有指标。

    Args:
        preds: 布尔列表或数组，True 表示预测匹配
        labels: 布尔列表或数组，True 表示真实匹配

    Returns:
        dict: {"precision": float, "recall": float, "f1": float, "accuracy": float}
    """
    # 转换为列表并确保长度一致
    preds = list(preds)
    labels = list(labels)
    assert len(preds) == len(labels), "preds and labels must have same length"

    counts = compute_confusion_counts(preds, labels)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    f1 = compute_f1(precision, recall)
    accuracy = compute_accuracy(tp, tn, fp, fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


def compute_confusion_counts(preds, labels):
    """Return TP/FP/FN/TN counts for boolean predictions and labels."""
    preds = list(preds)
    labels = list(labels)
    assert len(preds) == len(labels), "preds and labels must have same length"

    return {
        "tp": sum(1 for p, l in zip(preds, labels) if p and l),
        "fp": sum(1 for p, l in zip(preds, labels) if p and not l),
        "fn": sum(1 for p, l in zip(preds, labels) if not p and l),
        "tn": sum(1 for p, l in zip(preds, labels) if not p and not l),
    }
