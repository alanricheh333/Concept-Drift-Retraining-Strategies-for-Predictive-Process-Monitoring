
from core.prediction_methods.config.metrics.precision import Precision
from core.prediction_methods.config.metrics.recall import Recall


class F1Score:
    def __init__(self):
        pass

    def calculate(self, result):
        recall = Recall().calculate(result)
        precision = Precision().calculate(result)

        score = 2 * ((recall * precision) / (recall + precision))

        return score