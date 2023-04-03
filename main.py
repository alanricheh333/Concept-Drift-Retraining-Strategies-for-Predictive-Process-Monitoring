from enum import Enum
from numpy import NaN
from core.constants import SampleOption, SamplingMethod
import data
import core.prediction_methods.config.settings as pred
import core.prediction_methods.config.metrics as met
import core.prediction_methods.methods as methods
from core.prediction_methods.utils.logfile import LogFile
from core.concept_drift import drift_detection
import pm4py
from config import root_directory
import pandas as pd
import numpy as np
from subprocess import PIPE, Popen

from utils import import_csv_eventlog, import_xes_eventlog


class TrainMethod(Enum):
    SDL = "sdl"
    LIN = "lin"
    PASQUADIBISCEGLIE = "pasquadibisceglte"
    


def predict_after_detection(method: TrainMethod = TrainMethod.SDL, train_file:str = "sample", test_file:str = "sww_test"):

    if method is TrainMethod.SDL:
        #start with new data
        train_data = data.get_data(train_file)

        #select predection method
        settings = pred.SDL

        #select the percentage of training with is always 100%
        settings.train_percentage = 100
        #prepare data with predection method settings
        train_data.prepare(settings)

        #getting the prediction method parameters
        method = methods.get_prediction_method("SDL")

        
        #get the test data
        test_data = data.get_data(test_file)
        
        #select predection method for testing
        test_settings = pred.SDL
        
        #select the train percentage for test data which will always be 0%
        settings.train_percentage = 0
        #prepare test data for testing
        test_data.prepare(test_settings)

        #train the model with the training data
        model = method.train(train_data.train)

        #test model with the testing data
        results = method.test(model, test_data.test_orig)

        #calculate accuracy
        accuracy = met.ACCURACY.calculate(results)

    elif method is TrainMethod.LIN:
        #start with new data
        train_data = data.get_data(train_file)

        #select predection method
        settings = pred.LIN

        #select the percentage of training with is always 100%
        settings.train_percentage = 100
        #prepare data with predection method settings
        train_data.prepare(settings)

        #getting the prediction method parameters
        method = methods.get_prediction_method("LIN")

        
        #get the test data
        test_data = data.get_data(test_file)
        
        #select predection method for testing
        test_settings = pred.LIN
        
        #select the train percentage for test data which will always be 0%
        settings.train_percentage = 0
        #prepare test data for testing
        test_data.prepare(test_settings)

        #train the model with the training data
        model = method.train(train_data.train)

        #test model with the testing data
        results = method.test(model, test_data.test_orig)

        #calculate accuracy
        accuracy = met.ACCURACY.calculate(results)
    
    elif method is TrainMethod.PASQUADIBISCEGLIE:
         #start with new data
        train_data = data.get_data(train_file)

        #normalize date and time for train data
        for idx, row in train_data.logfile.data.iterrows():
            time = pd.to_datetime(row["completeTime"])
            train_data.logfile.data.at[idx, 'completeTime'] = time.strftime('%Y/%m/%d %H:%M:%S.%f')

        #select predection method
        settings = pred.PASQUADIBISCEGLIE

        #select the percentage of training with is always 100%
        settings.train_percentage = 100
        #prepare data with predection method settings
        train_data.prepare(settings)

        #getting the prediction method parameters
        method = methods.get_prediction_method("PASQUADIBISCEGLIE")

        
        #get the test data
        test_data = data.get_data(test_file)

        #nomalize the date and time for test data
        for idx, row in test_data.logfile.data.iterrows():
            time = pd.to_datetime(row["completeTime"])
            test_data.logfile.data.at[idx, 'completeTime'] = time.strftime('%Y/%m/%d %H:%M:%S.%f')
        
        #select predection method for testing
        test_settings = pred.PASQUADIBISCEGLIE
        
        #select the train percentage for test data which will always be 0%
        settings.train_percentage = 0
        #prepare test data for testing
        test_data.prepare(test_settings)

        #train the model with the training data
        model = method.train(train_data.train)

        #test model with the testing data
        results = method.test(model, test_data.test_orig)

        #calculate accuracy
        accuracy = met.ACCURACY.calculate(results)    
    
    
    
    print("accuracy: ", accuracy)
    


def predict():
    
    #data_object = data.get_data("IOR5k", time="time:timestamp", case="case:concept:name", activity="concept:name", resource="org:resource")
    #data_object = data.get_data("sample")
    data_object = data.get_data("sww_train")
    #data_object = data.get_data("sample_sdl")

    for idx, row in data_object.logfile.data.iterrows():
        dd = pd.to_datetime(row["completeTime"])
        data_object.logfile.data.at[idx, 'completeTime'] = dd.strftime('%Y/%m/%d %H:%M:%S.%f')
    # data_object.logfile.data.style.format({"completeTime": lambda t: t.strftime("%Y/%m/%d %H:%M:%S.%f")})
    # data_object.logfile.data["completeTime"] = pd.to_datetime(data_object.logfile.data["completeTime"])
    # data_object.logfile.data["completeTime"] = data_object.logfile.data["completeTime"].dt.strftime('%Y/%m/%d %H:%M:%S.%f')

    #settings = pred.LIN
    #settings = pred.SDL
    settings = pred.PASQUADIBISCEGLIE

    settings.train_percentage = 100
    data_object.prepare(settings)

    #m = methods.get_prediction_method("LIN")
    #m = methods.get_prediction_method("SDL")
    m = methods.get_prediction_method("PASQUADIBISCEGLIE")

    test_data = data.get_data("sww_test")

    for idx, row in test_data.logfile.data.iterrows():
        dd = pd.to_datetime(row["completeTime"])
        test_data.logfile.data.at[idx, 'completeTime'] = dd.strftime('%Y/%m/%d %H:%M:%S.%f')
    #test_settings = pred.LIN
    #test_settings = pred.SDL
    test_settings = pred.PASQUADIBISCEGLIE
    
    settings.train_percentage = 0
    test_data.prepare(test_settings)
    
    model = m.train(data_object.train)

    results = m.test(model, test_data.test_orig)

    accuracy = met.ACCURACY.calculate(results)

    print("accuracy: ", accuracy)


# def detect(customized_sampling: bool = False):
#     if customized_sampling:
#         #apply drift detection
#         all_sublogs = concept_drift.apply_drift_detection(log_name="sww_90", delta=0.002)
#         #apply general sampling
#         result_log = concept_drift.general_sampling(all_sublogs, probabilites=[1, 0, 0, 0], case_percent=0.6, all_cases=False)
#         # export result log
#         concept_drift.export_sampled_log(result_log)
#     else:
#         concept_drift.detect_drifts(use_saved_sublogs=False, sampling_method=concept_drift.SamplingMethod.UNIFORM, log_name="sww_90", over_sampling=True, all_cases=False)


def import_event_log():
    """
    Imports an event log then split it to train and test then export in csv and xes
    """
    #importing file settings
    file_path = "path_to_file"
    file_name = "some_name"
    split_percent = 90
    columns_names = {
        "case": "case",
        "activity": "activity",
        "timestamp": "timestamp",
        "resource": "resource"
    }
    naive = True
    xes = True

    if xes:
        result = import_xes_eventlog(file_path, file_name, split_percent, columns_names, naive)
    else: 
        result = import_csv_eventlog(file_path, file_name, split_percent, columns_names, naive)

    print(result)



def detect_drift_generate_sample_log():
    """
    Detects the drifts in a log then generate a sample log depending on a sample method and a sample option
    """
    drift_detection.get_sampled_log("sww_90", "100", True, 0.6, SamplingMethod.PRIORITY_LAST, SampleOption.CASES_FROM_COUNT_EVENTS)

if __name__ == '__main__':
    detect_drift_generate_sample_log()
    #predict()
    #detect()
    #predict_after_detection(method= TrainMethod.SDL)
