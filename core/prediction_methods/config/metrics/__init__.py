from metrics.accuracy import Accuracy
from metrics.cumm_accuracy import Cumm_Accuracy
from metrics.period_accuracy import Period_Accuracy
from metrics.brier import Brier
from metrics.precision import Precision
from metrics.recall import Recall

ACCURACY = Accuracy()
CUMM_ACCURACY = Cumm_Accuracy()
PERIOD_ACCURACY = Period_Accuracy(100)
BRIER = Brier()
PRECISION = Precision()
RECALL = Recall()