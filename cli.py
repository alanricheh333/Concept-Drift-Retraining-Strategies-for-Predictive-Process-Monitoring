
import os
from core.constants import PredictionMethod, SampleOption, SamplingMethod
from core.prediction_methods.predict import train_and_predict
from core.concept_drift import drift_detection
from config import root_directory
import click

@click.command()
@click.option("--original_log", help="Original log name, the train log")
@click.option("--test_log", help="The test log name")
@click.option("--sample_method", help="The sampling method to use, options are: uniform, only_last, priority_last, priority_first")
@click.option("--sample_option", help="The sampling option to use, options are: cases_events_count, cases_events, incomplete_cases, full_cases, only_events")
@click.option("--prediction_method", help="The prediction method to use, options are: mm-pred, cnn, sdl")
def predict_sampled_cli(original_log, test_log, sample_method, sample_option, prediction_method):
    """
    Train and test the next event prediction on the sampled log after detecting drifts
    """
    #check sampling method
    if sample_method == "uniform":
        sam_method = SamplingMethod.UNIFORM
    elif sample_method == "only_last":
        sam_method = SamplingMethod.ONLY_LAST
    elif sample_method == "priority_last":
        sam_method = SamplingMethod.PRIORITY_LAST
    elif sample_method == "priority_first":
        sam_method = SamplingMethod.PRIORITY_FIRST
    else:
        print("please pick a sampling method from the list: uniform, only_last, priority_last, priority_first !!")
        return

    #check sampling option
    if sample_option == "cases_events_count":
        sam_option = SampleOption.CASES_FROM_COUNT_EVENTS
    elif sample_option == "cases_events":
        sam_option = SampleOption.CASES_FROM_EVENTS
    elif sample_option == "incomplete_cases":
        sam_option = SampleOption.INCOMPLETE_CASES
    elif sample_option == "full_cases":
        sam_option = SampleOption.ONLY_FULL_CASES
    elif sample_option == "only_events":
        sam_option = SampleOption.ONLY_EVENTS
    else:
        print("please pick a sampling option from the list: cases_events_count, cases_events, incomplete_cases, full_cases, only_events !!")
        return

    #check prediction method
    if prediction_method == "mm-pred":
        pred_method = PredictionMethod.LIN
    elif prediction_method == "cnn":
        pred_method = PredictionMethod.PASQUADIBISCEGLIE
    elif prediction_method == "sdl":
        pred_method = PredictionMethod.SDL
    else:
        print("please pick a prediction method from the list: mm-pred, cnn, sdl !!")
        return


    original_train_log = original_log
    #detect drifts
    sampled_log = drift_detection.get_sampled_log(original_train_log, "100", True, 0.6, sam_method, sam_option)
    
    log_train = "sampled"
    log_test = test_log
    train_file_path = os.path.join(root_directory, "data", "output", log_train + ".csv")
    test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")
    
    accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path, pred_method)

    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1_score)


if __name__ == '__main__':
    predict_sampled_cli()