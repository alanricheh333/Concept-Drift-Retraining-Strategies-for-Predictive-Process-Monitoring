from .accuracy import Accuracy
from .cumm_accuracy import Cumm_Accuracy
from .period_accuracy import Period_Accuracy
from .brier import Brier
from .precision import Precision
from .recall import Recall

ACCURACY = Accuracy()
CUMM_ACCURACY = Cumm_Accuracy()
PERIOD_ACCURACY = Period_Accuracy(100)
BRIER = Brier()
PRECISION = Precision()
RECALL = Recall()