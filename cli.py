
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


@click.command()
@click.option("--original_log", help="Original log name, the train log")
@click.option("--test_log", help="The test log name")
@click.option("--sample_method", help="The sampling method to use, options are: uniform, only_last, priority_last, priority_first")
@click.option("--prediction_method", help="The prediction method to use, options are: mm-pred, cnn, sdl")
def predict_block_cli(original_log, test_log, sample_method, prediction_method):
    """
    Train and test the next event prediction on the sampled log after detecting drifts with all design options
    A Sampling method and a prediction method has to be specified
    """
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

    results = {}
    original_train_log = original_log
    sampling_options = [SampleOption.CASES_FROM_COUNT_EVENTS, SampleOption.INCOMPLETE_CASES, SampleOption.CASES_FROM_EVENTS, SampleOption.ONLY_FULL_CASES, SampleOption.ONLY_EVENTS]
    for sam in sampling_options:
    
        try:
            sam_option = sam

            #detect drifts
            sampled_log = drift_detection.get_sampled_log(original_train_log, "100", True, 0.6, sam_method, sam_option)

            print("########### SAMPLED LOG SIZE: ", len(sampled_log))

            log_train = "sampled"
            log_test = test_log
            train_file_path = os.path.join(root_directory, "data", "output", log_train + ".csv")
            test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")

            accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path, pred_method)
            single_result = {
                "accuracy": accuracy,
                "f1_score": f1_score
            }
            results[sam_option.value] = single_result
            print(sam_option.value, ": ",single_result)

        except Exception as error:
            print("ERROR: ", error)
            accuracy = ""
            f1_score = ""

    print("results: ", results)


@click.command()
@click.option("--original_log", help="Original log name, the train log")
@click.option("--test_log", help="The test log name")
@click.option("--prediction_method", help="The prediction method to use, options are: mm-pred, cnn, sdl")
def predict_larger_block_cli(original_log, test_log, prediction_method):
    """
    Train and test the next event prediction on the sampled log after detecting drifts with all design options
    A Sampling method and a prediction method has to be specified
    """
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


    results = {}
    original_train_log = original_log
    sampling_options = [SampleOption.CASES_FROM_COUNT_EVENTS, SampleOption.INCOMPLETE_CASES, SampleOption.CASES_FROM_EVENTS, SampleOption.ONLY_FULL_CASES, SampleOption.ONLY_EVENTS]
    sampling_method = [SamplingMethod.UNIFORM, SamplingMethod.PRIORITY_FIRST, SamplingMethod.PRIORITY_LAST, SamplingMethod.ONLY_LAST]

    for meth in sampling_method:
        method_result = {}
        
        for sam in sampling_options:
        
            try:
                sam_option = sam

                #detect drifts
                sampled_log = drift_detection.get_sampled_log(original_train_log, "100", True, 0.6, meth, sam_option)

                print("########### SAMPLED LOG SIZE: ", len(sampled_log))

                log_train = "sampled"
                log_test = test_log
                train_file_path = os.path.join(root_directory, "data", "output", log_train + ".csv")
                test_file_path = os.path.join(root_directory, "data", "input", "csv", log_test + ".csv")

                accuracy, f1_score = train_and_predict(log_train, train_file_path, log_test, test_file_path, pred_method)
                single_result = {
                    "accuracy": accuracy,
                    "f1_score": f1_score
                }
                method_result[sam_option.value] = single_result
                print(sam_option.value, ": ",single_result)

            except Exception as error:
                print("ERROR: ", error)
                accuracy = ""
                f1_score = ""

        results[meth.value] = method_result

    print("results: ", results)


@click.group()
def cli():
    pass


cli.add_command(predict_sampled_cli)
cli.add_command(predict_block_cli)
cli.add_command(predict_larger_block_cli)


if __name__ == '__main__':
    cli()
