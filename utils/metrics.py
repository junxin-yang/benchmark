import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


# Classification Metrics
def acc(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return round(accuracy_score(y_trues, pred_classes), 2)


def precision(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return round(precision_score(y_trues, pred_classes, average='macro', zero_division=0), 2)


def recall(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return round(recall_score(y_trues, pred_classes, average='macro', zero_division=0), 2)


def f1(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return round(f1_score(y_trues, pred_classes, average='macro', zero_division=0), 2)


def auc(y_trues, y_preds_proba):
    y_trues = np.array(y_trues)
    y_preds_proba = np.array([y_pred.get('probabilities') for y_pred in y_preds_proba])
    # binary classification
    if y_preds_proba.ndim == 2 and y_preds_proba.shape[1] == 2:
        # only take positive class probability
        y_preds_proba = y_preds_proba[:, 1]
        return round(roc_auc_score(y_trues, y_preds_proba), 2)
    # multi-class
    elif y_preds_proba.ndim == 2 and y_preds_proba.shape[1] > 2:
        return round(roc_auc_score(y_trues, y_preds_proba, multi_class='ovr'), 2)
    # single-class (degenerate case)
    elif y_preds_proba.ndim == 1:
        return round(roc_auc_score(y_trues, y_preds_proba), 2)
    else:
        raise ValueError(f"Unexpected y_preds_proba shape: {y_preds_proba.shape}")


# Report Generation Metric
def bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    计算BLEU分数

    :param references: list of list of reference sentences (每个参考可以是多个)
                        e.g., [[ref1_tokens], [ref2_tokens], ...]
    :param hypotheses: list of hypothesis sentences (模型生成的句子)
                        e.g., [hypothesis_tokens1, hypothesis_tokens2, ...]
    :param weights: BLEU权重，默认是4-gram均匀加权
    :return: BLEU分数 (0-1)
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        # ref需要是list of list，hyp是list
        if isinstance(ref[0], str):
            ref = [ref]  # 只有一个参考时包装一下
        score = sentence_bleu(ref, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return round(sum(scores) / len(scores), 2)  # 平均BLEU分数
    

# Survival Analysis Metrics
def c_index(y_true, y_preds):
    """
    y_true: [(time, event), ...]  event=1表示终点事件发生，0表示截尾
    y_pred: 风险分数或生存概率，越大风险越高
    """
    times = [t[0] for t in y_true]
    events = [t[1] for t in y_true]
    risk_preds = [y_pred.get('risk_score') if isinstance(y_pred, dict) else y_pred for y_pred in y_preds]
    return round(concordance_index(times, risk_preds, events), 2)


def auc_survival(y_true, y_preds):
    """
    y_true: [(time, event), ...] 只用event作为标签
    y_pred: 风险分数或生存概率
    """
    events = [t[1] for t in y_true]
    survival_preds = [y_pred.get('risk_score') if isinstance(y_pred, dict) else y_pred for y_pred in y_preds]
    return round(roc_auc_score(events, survival_preds), 2)