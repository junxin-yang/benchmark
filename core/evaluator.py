# metrics.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Metrics:
    def __init__(self):
        self.smooth = SmoothingFunction().method1  # 平滑函数，避免BLEU为0
    
    def bleu(self, references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)):
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
            score = sentence_bleu(ref, hyp, weights=weights, smoothing_function=self.smooth)
            scores.append(score)
        return sum(scores) / len(scores)  # 平均BLEU分数

# 使用示例
if __name__ == "__main__":
    metric = Metrics()
    
    references = [
        [["this", "is", "a", "test"]],
        [["hello", "world"]],
    ]
    hypotheses = [
        ["this", "is", "test"],
        ["hello", "word"]
    ]
    
    bleu_score = metric.bleu(references, hypotheses)
    print("BLEU score:", bleu_score)
