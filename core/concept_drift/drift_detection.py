import pandas as pd

from core.concept_drift.classes.sampling_methods import OnlyLastSampling, PrioritySampling, UniformSampling
from core.concept_drift.classes.sampling_options import CasesBasedEventsCountSamples, CasesFromEventsSamples, EventsSamples, IncompleteCasesSamples, OnlyFullCasesSamples
from core.constants import TIMESTAMP_IDENTIRIFIER_CSV, SampleOption, SamplingMethod
from .classes.sublog import SubLog
import os
from config import root_directory
import numpy as np
from . import utils
import math
from subprocess import check_output


def get_sampled_log(event_log: str, window_size: str = "100", all_cases: bool = True, cases_percentage: float = 0.6,
                    sampling_method: SamplingMethod = SamplingMethod.ONLY_LAST, sampling_option: SampleOption = SampleOption.ONLY_FULL_CASES) -> pd.DataFrame:
    """
    Detect the drifts then construct the sampled log depending on a sampling method for buckets and a sampling option for cases
    Params - event_log: the name of the training event log
             window_size: the window size chosen for the drift detection method
             all_cases: if all cases number should be sampled
             cases_percentage: if a portion of the number of cases is desired not the whole number
             sampling_method: the sampling method chosen to sample buckets
             sampling_option: the sampling option chosen to sample cases from buckets

    Retruns - the sampled log as a dataframe
    """
    
    #detect the drift and divide into sublogs
    all_sublogs = detect_drifts(event_log=event_log, window_size=window_size)

    #choose a specific buckets sampling technique
    #choice between: PRIORITY_LAST, PRIORITY_FIRST, UNIFORM and ONLY_LAST
    if sampling_method == SamplingMethod.PRIORITY_LAST:
        priority_sampling = PrioritySampling()
        sublogs_with_weights = priority_sampling.apply_priority_sampling(all_sublogs, True)

    if sampling_method == SamplingMethod.PRIORITY_FIRST:
        priority_sampling = PrioritySampling()
        sublogs_with_weights = priority_sampling.apply_priority_sampling(all_sublogs, False)
    
    elif sampling_method == SamplingMethod.UNIFORM:
        uniform_sampling = UniformSampling()
        sublogs_with_weights = uniform_sampling.apply_uniform_sampling(all_sublogs)

    elif sampling_method == SamplingMethod.ONLY_LAST:
        only_last_sampling = OnlyLastSampling()
        sublogs_with_weights = only_last_sampling.apply_only_last_sampling(all_sublogs)

    
    #get the number of cases in the training log
    original_log = pd.read_csv(os.path.join(root_directory, "data", "input", "csv", event_log + ".csv"))
    cases_number = len(original_log["case"].unique())
    
    #if consider all cases then the sample size should be the same as the whole cases size
    if all_cases:
        sample_size = cases_number
    #else the sample size should be a precentage of the cases size
    else:
        sample_size = math.ceil(cases_number * cases_percentage)

    #sample buckets then cases from buckets and construct a sampled log
    sampled_log = construct_sample_log(sublogs_with_weights, sample_size, sampling_option)

    sampled_log = equalize_activities_num(sampled_log, event_log)
    
    #export the sampled log
    utils.export_sampled_log(sampled_log)
    
    return sampled_log


def equalize_activities_num(sampled_log:pd.DataFrame, log_name:str):

    test_file_path = os.path.join(root_directory, "data", "input", "csv", log_name.replace("_train", "_test") + ".csv")
    test = pd.read_csv(test_file_path, index_col=0)
    train_act = sampled_log["event"].unique()
    test_act = test["event"].unique()

    print(len(train_act))
    print(len(test_act))
    dif = list(set(train_act) - set(test_act)) + list(set(test_act) - set(train_act))
    print("DIF: ", dif)
    #test = test.drop(columns=["Unnamed: 0"])
    
    if len(train_act) > len(test_act):
        for act in dif:
            last = test.tail(1)
            new_index = len(test)
            last["event"] = act
            new_data = pd.DataFrame(last.values, index=[new_index], columns=test.columns)
            test = pd.concat([test, new_data])
            print(test.tail(1))
        
        test.to_csv(test_file_path)
   
    elif len(train_act) < len(test_act):
        for act in dif:
            last = sampled_log.tail(1)
            new_index = len(sampled_log)
            last["event"] = act
            new_data = pd.DataFrame(last.values, index=[new_index], columns=sampled_log.columns)
            sampled_log = pd.concat([sampled_log, new_data])
            print(sampled_log.tail(1))
        
    
    return sampled_log



def detect_drifts(event_log: str, window_size: str = "100") -> list[SubLog]:
    """
    Detect the concept drifts in the log.
    Uses ProDrift tool to detect the drifts
    Params - event_log: the event log name
             window_size: the size of the adaptive window used to detect the drifts

    Returns - a list of Sublog object
    """
    
    #open process to connect with the pro drift jar file
    output = check_output(['java', '-jar', os.path.join(root_directory, "core", "concept_drift", "ProDrift2.5.jar"), '-fp', 
                     os.path.join(root_directory, "data", "input", "xes", event_log + ".xes"),
                     '-ddm','events', '-ws', window_size, '-ddnft', '0.0', '-dds', 'high', '-cm', 'activity', '-dcnft', '0.0'], text=True)
    
    #get the result as text
    result_text = output
    print(result_text)
    #retrieve the useful information from the resulting text
    first_split:list = result_text.split('\n\n\n')
    second_split:list = first_split[0].split('\n\n')
    drift_text_list = []
    drift_text_list.append(second_split[1])
    first_split.pop(0)
    first_split.remove('')
    drift_text_list.extend(first_split)

    #restructre the retrieved information into dict with the event id as key and the event timestamp as value
    drifts_dict = {}
    for item in drift_text_list:
        important_text:str = item.partition("Drift detected at event: ")[2]
        data_info = important_text.split(" ", 1)
        event_num = int(data_info[0])
        event_date = data_info[1][ data_info[1].find("(")+1 : data_info[1].find(")") ]
        drifts_dict[event_num] = event_date
    
    #read the event log in csv format
    dataset = pd.read_csv(os.path.join(root_directory, "data", "input", "csv", event_log + ".csv"))
    
    #split the event log into sublogs depending on the drifts detected
    old_index = 0
    sublogs = []
    for key, value in drifts_dict.items():
        sub = dataset[old_index:key]
        sublogs.append(sub)
        old_index = key+1

    #append list sublog
    sub = dataset[old_index:len(dataset)]
    sublogs.append(sub)
    
    #create a Sublog object for each sublog created
    all_sublogs:list[SubLog] = []
    for sublog in sublogs:
        #TODO: add if needed to add the xes sublog
        # sublog = sublog.rename(columns={'case':'case:concept:name', 'role':'org:resource', 'event':'concept:name', 'completeTime':'time:timestamp'})
        # xes_log = pm4py.convert_to_event_log(sublog)
        # log = EventLog(xes_log)
        result_sublog = SubLog(from_case=0, to_case=len(sublog), sub_log=None, event_log=sublog)
        all_sublogs.append(result_sublog)

    print(len(all_sublogs))
    return all_sublogs



def construct_sample_log(sublogs: list[SubLog], sample_size: int, sampling_option: SampleOption = SampleOption.ONLY_FULL_CASES) -> pd.DataFrame:
    """
    Samples the buckets depending on the given weight of each bucket then sample the cases depending on the sampling option
    Params - sublogs: the list of sublogs returned after drift detection
             sample_number: the size of sample i.e. the number of cases to be sampled from the buckets
             sampling_option: the sampling option chosen to sample cases from buckets

            
    Returns - the sampled log as a dataframe
    """

    probabilites = []
    # get the probabilities/weights of each sublog
    for sublog in sublogs:
        probabilites.append(sublog.weight)

    # get the number of buckets(sublogs)
    buckets = np.arange(len(sublogs))

    # sample the buckets depending on their probabilities and the number of samples needed
    sampled_buckets = np.random.choice(
        buckets, p=probabilites, size=int(sample_size))

    # generate the sampled cases depending on the buckets probablities
    sampled_cases = sample_cases_from_buckets(sampled_buckets, sublogs, sampling_option)

    # combine the list of dataframes(sampled cases) in one dataframe
    event_log: pd.DataFrame = pd.concat(sampled_cases)
    # sort by time
    event_log = event_log.sort_values(TIMESTAMP_IDENTIRIFIER_CSV)

    return event_log



def sample_cases_from_buckets(sampled_buckets: list[int], sublogs: list[SubLog], sampling_option: SampleOption = SampleOption.ONLY_FULL_CASES) -> list[SubLog]:
    """
    Sample list of buckets depending on buckets probabilites and sample number
    Params - sampled_buckets: list of int representing the indecies of sampled buckets
             sublogs: list of sublogs constructed by concept drift detection method
             sampling_option: the sampling option chosen to sample cases from buckets

    Returns - list of sampled cases
    """

    list_of_sampled_cases = []
    
    #choose the sampling option and sample cases depending on the sampling option
    #choice between: ONLY_FULL_CASES, CASES_FROM_EVENTS, CASES_FROM_COUNT_EVENTS, INCOMPLETE_CASES, ONLY_EVENTS
    if sampling_option == SampleOption.ONLY_FULL_CASES:
        technique = OnlyFullCasesSamples()
        list_of_sampled_cases = technique.sample_only_full_cases_from_buckets(sublogs, sampled_buckets)

    if sampling_option == SampleOption.CASES_FROM_EVENTS:
        technique = CasesFromEventsSamples()
        list_of_sampled_cases = technique.sample_cases_based_on_sampling_events(sublogs, sampled_buckets)

    if sampling_option == SampleOption.CASES_FROM_COUNT_EVENTS:
        technique = CasesBasedEventsCountSamples()
        list_of_sampled_cases = technique.sample_cases_based_on_events_count(sublogs, sampled_buckets)

    if sampling_option == SampleOption.INCOMPLETE_CASES:
        technique = IncompleteCasesSamples()
        list_of_sampled_cases = technique.sample_incomplete_cases(sublogs, sampled_buckets)

    if sampling_option == SampleOption.ONLY_EVENTS:
        technique = EventsSamples()
        list_of_sampled_cases = technique.sample_only_events(sublogs, sampled_buckets)

    
    return list_of_sampled_cases



