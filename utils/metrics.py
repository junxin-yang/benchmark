import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


# Classification Metrics
def acc(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return accuracy_score(y_trues, pred_classes)


def precision(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return precision_score(y_trues, pred_classes, average='macro', zero_division=0)


def recall(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return recall_score(y_trues, pred_classes, average='macro', zero_division=0)


def f1(y_trues, y_preds):
    pred_classes = [y_pred.get('pred_class') for y_pred in y_preds]
    return f1_score(y_trues, pred_classes, average='macro', zero_division=0)


def auc(y_trues, y_preds_proba):
    y_trues = np.array(y_trues)
    y_preds_proba = np.array([y_pred.get('probabilities') for y_pred in y_preds_proba])
    # binary classification
    if y_preds_proba.ndim == 2 and y_preds_proba.shape[1] == 2:
        # only take positive class probability
        y_preds_proba = y_preds_proba[:, 1]
        return roc_auc_score(y_trues, y_preds_proba)
    # multi-class
    elif y_preds_proba.ndim == 2 and y_preds_proba.shape[1] > 2:
        return roc_auc_score(y_trues, y_preds_proba, multi_class='ovr')
    # single-class (degenerate case)
    elif y_preds_proba.ndim == 1:
        return roc_auc_score(y_trues, y_preds_proba)
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
    return sum(scores) / len(scores)  # 平均BLEU分数
    

# Survival Analysis Metrics
def c_index(y_true, y_preds):
    """
    y_true: [(time, event), ...]  event=1表示终点事件发生，0表示截尾
    y_pred: 风险分数或生存概率，越大风险越高
    """
    times = [t[0] for t in y_true]
    events = [t[1] for t in y_true]
    risk_preds = [y_pred.get('risk_score') if isinstance(y_pred, dict) else y_pred for y_pred in y_preds]
    return concordance_index(times, risk_preds, events)


def auc_survival(y_true, y_preds):
    """
    y_true: [(time, event), ...] 只用event作为标签
    y_pred: 风险分数或生存概率
    """
    events = [t[1] for t in y_true]
    survival_preds = [y_pred.get('risk_score') if isinstance(y_pred, dict) else y_pred for y_pred in y_preds]
    return roc_auc_score(events, survival_preds)


if __name__ == "__main__":
    # 简单测试
    all_labels = [0, 2, 1, 0, 2]
    all_preds = [0, 2, 1, 0, 2]
    metrics = [acc, precision, recall, f1]
    metric_results = {}

    for metric_fn in metrics:
        metric_name = metric_fn.__name__
        metric_results[metric_name] = metric_fn(all_labels, all_preds)
        print(f"{metric_name}: {metric_results[metric_name]}")


    # 测试AUC
    all_labels = [0, 2, 1, 0, 2]
    all_preds_proba = [
        [0.8, 0.1, 0.1],  # 样本1对3类的概率
        [0.2, 0.2, 0.6],
        [0.1, 0.7, 0.2],
        [0.05, 0.9, 0.05],
        [0.05, 0.9, 0.05]
    ]
    print(f"{auc.__name__}: {auc(all_labels, all_preds_proba)}")
    
    references = [
        [["this", "is", "a", "test"]],
        [["hello", "world"]],
    ]
    hypotheses = [
        ["this", "is", "test"],
        ["hello", "word"]
    ]
    
    print(f"{bleu.__name__}: {bleu(references, hypotheses)}")

    # 假设有5个样本
    # y_true 组织为 [(time, event), ...]
    y_true = [
        (10, 1),  # 第1个样本，生存时间10，事件发生
        (20, 0),  # 第2个样本，生存时间20，截尾
        (30, 1),  # 第3个样本，生存时间30，事件发生
        (40, 1),  # 第4个样本，生存时间40，事件发生
        (50, 0)   # 第5个样本，生存时间50，截尾
    ]
    # y_pred 是每个样本的风险分数或生存概率（越大风险越高）
    y_pred = [0.2, 0.8, 0.4, 0.6, 0.3]

    # 计算C-index
    print(f"{c_index.__name__}: {c_index(y_true, y_pred)}")

    # 计算AUC（只用event作为标签）
    print(f"{auc_survival.__name__}: {auc_survival(y_true, y_pred)}")