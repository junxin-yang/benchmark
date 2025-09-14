from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def plot_roc_curve(self, labels, probs):
    # 实现ROC曲线绘制
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(labels, probs, ax=ax)
    ax.set_title('ROC Curve')
    return fig
    
def plot_confusion_matrix(self, labels, preds):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(labels, preds, ax=ax)
    ax.set_title('Confusion Matrix')
    return fig