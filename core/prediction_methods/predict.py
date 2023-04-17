from typing import Tuple
from core.constants import PredictionMethod
import core.prediction_methods.config.settings as predictor
from core.prediction_methods.config.settings.setting import Setting
import core.prediction_methods.methods as methods
import core.prediction_methods.config.metrics as metric
import data
from core.constants import TIMESTAMP_IDENTIRIFIER_CSV
import pandas as pd


def train_and_predict(train_log_name: str, train_log_path: str, test_log_name: str, test_log_path: str, prediction_method: PredictionMethod = PredictionMethod.SDL) -> Tuple[float, float]:
    """
    Train and predict 
    Params - train_log_name: the name of the train log
             train_log_path: the path of the train log
             test_log_name: the name of the test log
             test_log_path: the path of the test log
             prediction_method: the prediction method selected

    Returns - the accuracy and f1-score metrics
    """

    # get the training data
    train_data = data.get_data(train_log_name, train_log_path)

    # get the settings of the prediction method
    settings, test_settings = get_model_settings(prediction_method)

    # get the test data
    test_data = data.get_data(test_log_name, test_log_path)

    # change the timestamp format for this type of prediction
    if prediction_method.value in PredictionMethod.PASQUADIBISCEGLIE.value:

        train_data.logfile.data[TIMESTAMP_IDENTIRIFIER_CSV] = pd.to_datetime(
            train_data.logfile.data[TIMESTAMP_IDENTIRIFIER_CSV]).strftime('%Y/%m/%d %H:%M:%S.%f')

        test_data.logfile.data[TIMESTAMP_IDENTIRIFIER_CSV] = pd.to_datetime(
            test_data.logfile.data[TIMESTAMP_IDENTIRIFIER_CSV]).strftime('%Y/%m/%d %H:%M:%S.%f')

    # select the percentage of training which is always 100%
    settings.train_percentage = 100
    # prepare data with predection method settings
    train_data.prepare(settings)

    # select the train percentage for test data which will always be 0%
    settings.train_percentage = 0
    # prepare test data for testing
    test_data.prepare(test_settings)

    # getting the prediction method parameters
    method = methods.get_prediction_method(prediction_method.value)

    # train the model with the training data
    model = method.train(train_data.train)

    # test model with the testing data
    results = method.test(model, test_data.test_orig)

    results_list = list(results)

    # calculate accuracy
    accuracy = metric.ACCURACY.calculate(results_list)

    # calculate f1-score
    f1_score = metric.F1SCORE.calculate(results_list)

    return accuracy, f1_score


def get_model_settings(prediction_method: PredictionMethod) -> Tuple[Setting, Setting]:
    """
    Gets the ml model settings depending on the prediction method
    Params - prediction_method: the used prediction method

    Returns - the settings for the train and test settings
    """
    if prediction_method.value is PredictionMethod.LIN.value:
        # select predection method
        settings = predictor.LIN

        # select predection method for testing
        test_settings = predictor.LIN

    elif prediction_method.value in PredictionMethod.SDL.value:
        # select predection method
        settings = predictor.SDL

        # select predection method for testing
        test_settings = predictor.SDL

    elif prediction_method.value in PredictionMethod.PASQUADIBISCEGLIE.value:
        # select predection method
        settings = predictor.PASQUADIBISCEGLIE

        # select predection method for testing
        test_settings = predictor.PASQUADIBISCEGLIE

    else:
        raise Exception("There is no prediction method")

    return settings, test_settings
