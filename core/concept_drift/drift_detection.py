from enum import Enum
from subprocess import PIPE, Popen
import pandas as pd

from core.concept_drift.classes.sampling_methods import OnlyLastSampling, PrioritySampling, UniformSampling
from .classes.sublog import SubLog
from pm4py.objects.log.obj import EventLog
import pm4py
import os
from config import root_directory
import numpy as np
from . import utils
import math



class SamplingMethod(Enum):
    UNIFORM = "Unifrom"
    PRIORITY_LAST = "Priority Last"
    PRIORITY_FIRST = "Priority First"
    ONLY_LAST = "Only Last"


class SampleOption(Enum):
    CASES_FROM_EVENTS = "Cases depending on events"
    ONLY_FULL_CASES = "Only full cases"
    CASES_FROM_COUNT_EVENTS = "Cases depending on the count of events"
    INCOMPLETE_CASES = "Cases even if incomplete"
    ONLY_EVENTS = "Not Cases just Events"


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
    original_log = pd.read_csv(os.path.join(os.getcwd(), "data", "input", "csv", event_log + ".csv"))
    cases_number = len(original_log["case"].unique())
    
    #if consider all cases then the sample number should be the same as the cases number
    if all_cases:
        sample_num = cases_number
    #else the sample number should be a precentage of the cases number
    else:
        sample_num = math.ceil(cases_number * cases_percentage)

    #sample buckets then cases from buckets and construct a sampled log
    sampled_log = sample_data(sublogs_with_weights, sample_num)
    
    #export the sampled log
    utils.export_sampled_log(sampled_log)
    
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
    process = Popen(['java', '-jar', os.path.join(root_directory, "core", "concept_drift", "ProDrift2.5.jar"), '-fp', 
                     os.path.join(root_directory, "data", "input", "xes", event_log + ".xes"),
                     '-ddm','events', '-ws', window_size, '-ddnft', '0.0', '-dds', 'high', '-cm', 'activity', '-dcnft', '0.0'], stdout=PIPE, stderr=PIPE)
    
    #excute the process
    result = process.communicate()
    
    #get the result as text
    result_text:str = result[0].decode('utf-8')

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
        sublog = sublog.rename(columns={'case':'case:concept:name', 'role':'org:resource', 'event':'concept:name', 'completeTime':'time:timestamp'})
        xes_log = pm4py.convert_to_event_log(sublog)
        log = EventLog(xes_log)
        result_sublog = SubLog(from_case=0, to_case=len(log._list), sub_log=log, event_log=sublog)
        all_sublogs.append(result_sublog)

    print(len(all_sublogs))
    return all_sublogs



def sample_data(sublogs: list[SubLog], sample_number: int) -> pd.DataFrame:
    """
    Samples the buckets depending on the given weight of each bucket then sample the cases depending on the sampling option
    Params - sublogs: the list of sublogs returned after drift detection
             sample_number: the number of cases to be sampled from the buckets
            
    Returns - the sampled log as a dataframe
    """

    probabilites = []
    # get the probabilities/weights of each sublog
    for sublog in sublogs:
        probabilites.append(sublog.weight)

    # get the number of buckets(sublogs)
    buckets = np.arange(len(sublogs))

    # generate the sampled cases depending on the bucksets probablities
    sampled_cases = sample_bucket_based_on_probability(
        buckets, probabilites, sample_number, sublogs)

    # create an event log and return it
    result_log = EventLog(sampled_cases)

    # turn to dataframe so it can be sorted by time
    dataframe = pm4py.convert_to_dataframe(result_log)
    dataframe = dataframe.sort_values("time:timestamp")
    result_log = pm4py.convert_to_event_log(dataframe)

    return result_log


def sample_bucket_based_on_probability(buckets: list[int], probabilites: list[int], all_sample_num: int, sublogs: list[SubLog],
                                        sampling_option: SampleOption = SampleOption.ONLY_FULL_CASES) -> list[SubLog]:
    """
    Sample list of buckets depending on buckets probabilites and sample number
    Params - buckets: list of buckets numbers
             probabilites: list of buckets probabilites
             all_sample_num: number of the samples
             sublogs: list of sublogs constructed by concept drift detection method
    Returns - list of sampled cases
    """

    # sample the buckets depending on their probabilities and the number of samples needed
    sampled_buckets = np.random.choice(
        buckets, p=probabilites, size=int(all_sample_num))

    list_of_sampled_cases = []
    
    #get unique cases for each bucket
    bucket_cases = {}
    for idx, sublog in enumerate(sublogs):
        bucket_cases[idx] = sublog.event_log["case:concept:name"].unique()

    # sample a case from each time a certain bucket accure
    for bucket in sampled_buckets:
        sublog: SubLog = sublogs.__getitem__(bucket)

        if sampling_option == SampleOption.ONLY_FULL_CASES:
            
            #sample a case id from the dedicated bucket
            sampled_case_id = np.random.choice(bucket_cases[bucket])
            is_in_another_bucket = False
            
            for key, value in bucket_cases.items():
               #check if the case is in other buckets
               if sampled_case_id in value and key != bucket:
                   is_in_another_bucket = True
                   break
            
            #if the case is in other buckets then continue and don't take the case
            if is_in_another_bucket:
                continue

        #sample the case from the sublog
        sampled_case = sublog.event_log.loc[sublog.event_log["case:concept:name"] == sampled_case_id]
        
        list_of_sampled_cases.append(sampled_case)

    return list_of_sampled_cases



