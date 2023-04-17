
import core.prediction_methods.config.metrics as metric
from core.prediction_methods.config.metrics.precision import Precision
from core.prediction_methods.config.metrics.recall import Recall


class F1Score:
    def __init__(self):
        pass

    def calculate(self, result):
        recall = metric.RECALL.calculate(result)
        precision = metric.PRECISION.calculate(result)

        try:
            score = 2 * ((recall * precision) / (recall + precision))
        except:
            score = 0

        return score