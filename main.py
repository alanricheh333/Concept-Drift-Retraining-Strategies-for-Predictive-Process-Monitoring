import os
from core.constants import SampleOption, SamplingMethod
from core.prediction_methods.predict import train_and_predict
from core.concept_drift import drift_detection
from config import root_directory
from utils import import_csv_eventlog, import_xes_eventlog


def predict_original():
    """
    Train and test the next event prediction on the original log without detecting drifts
    """
    log_train = "sww_train"
    log_test = "sww_test"
    train_file_path = os.path.join(root_directory, "data", "input", "csv", log_train + ".csv")
    test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")
    
    accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path)

    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1_score)
    

def predict_sampled():
    """
    Train and test the next event prediction on the sampled log after detecting drifts
    """
    original_train_log = "sww_train"
    #detect drifts
    sampled_log = drift_detection.get_sampled_log(original_train_log, "100", True, 0.6, SamplingMethod.PRIORITY_LAST, SampleOption.CASES_FROM_COUNT_EVENTS)
    
    log_train = "sampled"
    log_test = "sww_test"
    train_file_path = os.path.join(root_directory, "data", "output", log_train + ".csv")
    test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")
    
    accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path)

    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1_score)


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
    #detect_drift_generate_sample_log()
    #import_event_log()
    #predict_sampled()
    predict_original()
