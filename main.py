import os
from core.concept_drift.classes.decay_sampling import DecaySampling
from core.constants import CASE_IDENTIFIER_CSV, TIMESTAMP_IDENTIRIFIER_XES, PredictionMethod, SampleOption, SamplingMethod
import core.prediction_methods.config.metrics as metric
from core.prediction_methods.predict import train_and_predict
from core.concept_drift import drift_detection
from config import root_directory
from data.utils import import_csv_eventlog, import_xes_eventlog
import pm4py
import pandas as pd

from experiments.visualize import plot_bar_chart, plot_drifts, plot_line_chart


def predict_original():
    """
    Train and test the next event prediction on the original log without detecting drifts
    """
    log_train = "gradual_log_noise_train"
    log_test = "gradual_log_noise_test"
    train_file_path = os.path.join(root_directory, "data", "input", "csv", log_train + ".csv")
    test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")
    
    accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path, PredictionMethod.PASQUADIBISCEGLIE)

    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1_score)
    

def predict_sampled():
    """
    Train and test the next event prediction on the sampled log after detecting drifts
    """
    original_train_log = "incremental_log_10000_train"
    #detect drifts
    sampled_log = drift_detection.get_sampled_log(original_train_log, "100", True, 0.6, SamplingMethod.ONLY_LAST, SampleOption.INCOMPLETE_CASES)
    
    log_train = "sampled"
    log_test = "incremental_log_10000_test"
    train_file_path = os.path.join(root_directory, "data", "output", log_train + ".csv")
    test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")
    
    accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path, PredictionMethod.PASQUADIBISCEGLIE)

    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1_score)


def import_event_log():
    """
    Imports an event log then split it to train and test then export in csv and xes
    """
    #importing file settings
    file_path = "/Users/alanalrechah/Desktop/Uni/Thesis/Code/Concept-Drift-Retraining-Strategies-for-Predictive-Process-Monitoring/data/input/csv/Helpdesk.csv"
    file_name = "helpdesk"
    split_percent = 90
    # columns_names = {
    #     "case": "case:concept:name",
    #     "activity": "concept:name",
    #     "timestamp": "time:timestamp",
    #     "resource": "org:resource"
    # }
    columns_names = {
        "case": "case",
        "activity": "event",
        "timestamp": "completeTime",
        "resource": "role"
    }
    naive = True
    xes = False

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


def plot_charts():
    """
    Plot both bar and line charts
    """
    bar = True
    path = os.path.join(root_directory, "experiments", "datasets", "recurring_sdl_plot.csv")
    dataset = pd.read_csv(path)

    if bar:
        plot_bar_chart(dataset)
    else:
        plot_line_chart(dataset)


def plot_drift_points():
    """
    Plot drifts in time
    """
    event_log = "helpdesk_train"

    plot_drifts(event_log)


def generate_synth_logs():
    """
    Generates synthetic logs
    """
    from conceptdrift.drifts.gradual import generate_log_with_gradual_drift
    from conceptdrift.drifts.sudden import generate_log_with_sudden_drift
    from conceptdrift.drifts.recurring import generate_log_with_recurring_drift
    from conceptdrift.drifts.incremental import generate_log_with_incremental_drift

    #event_log = generate_log_with_sudden_drift(num_traces=50000, change_point=0.7, change_proportion=0.2)
    #event_log = generate_log_with_gradual_drift(num_traces=10000, start_point=0.4, end_point=0.92, distribution_type="linear", change_proportion = 0.2)
    #event_log = generate_log_with_recurring_drift(num_traces=10000, start_point=0.4, end_point=0.92)
    event_log = generate_log_with_incremental_drift(num_versions=4, traces=[700, 2766, 3244, 7400], change_proportion=0.2)

    df = pm4py.convert_to_dataframe(event_log)
    df["lifecycle:transition"] = "complete"
    df["org:resource"] = ""
    
    event_log = pm4py.convert_to_event_log(df)
    pm4py.write_xes(event_log, os.path.join(root_directory, "data", "input", "xes", "incremental_log_10000.xes"))


def sample_decay_log():
    """
    Samples cases from decay weighted sampling
    """
    path = os.path.join(root_directory, "data", "input", "csv", "helpdesk_train.csv")
    log = pd.read_csv(path, index_col=0)

    weighted_log = DecaySampling.apply_exponential_decay(log, 0.2)

    sample_num = len(weighted_log[CASE_IDENTIFIER_CSV].unique())
    sampled_log = DecaySampling.sample_cases_original_log(weighted_log, sample_num)

    sampled_log.to_csv(os.path.join(root_directory, "data", "output", "decay_sampled_helpdesk.csv"))

    sampled_log, test_log = DecaySampling.equalize_activities_num(os.path.join(root_directory, "data", "output", "decay_sampled_helpdesk.csv"), 
                                                                  os.path.join(root_directory, "data", "input", "csv", "helpdesk_test.csv"))




if __name__ == '__main__':
    #detect_drift_generate_sample_log()
    #import_event_log()
    #predict_sampled()
    #predict_original()
    plot_charts()
    #plot_drift_points()
    #generate_synth_logs()
    #sample_decay_log()  

    
