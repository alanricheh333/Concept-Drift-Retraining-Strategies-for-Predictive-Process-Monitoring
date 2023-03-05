from subprocess import PIPE, Popen
import pandas as pd
from .classes.sublog import SubLog
from pm4py.objects.log.obj import EventLog
import pm4py
import os



def detect_drifts(event_log: str) -> list[SubLog]:
    
    ss = os.path.join(os.getcwd(), "core", "concept_drift", "ProDrift2.5.jar")
    aa = os.path.join(os.getcwd(), "data", "input", event_log+".xes")

    process = Popen(['java', '-jar', os.path.join(os.getcwd(), "core", "concept_drift", "ProDrift2.5.jar"), '-fp', os.path.join(os.getcwd(), "data", "input", event_log + ".xes"),
                     '-ddm','events', '-ws', '100', '-ddnft', '0.0', '-dds', 'high', '-cm', 'activity', '-dcnft', '0.0'], stdout=PIPE, stderr=PIPE)
    
    result = process.communicate()
    
    result_text:str = result[0].decode('utf-8')

    first_split:list = result_text.split('\n\n\n')
    second_split:list = first_split[0].split('\n\n')
    
    drift_text_list = []
    drift_text_list.append(second_split[1])
    first_split.pop(0)
    first_split.remove('')
    drift_text_list.extend(first_split)

    drifts_dict = {}
    for item in drift_text_list:
        important_text:str = item.partition("Drift detected at event: ")[2]
        data_info = important_text.split(" ", 1)
        event_num = int(data_info[0])
        event_date = data_info[1][ data_info[1].find("(")+1 : data_info[1].find(")") ]
        drifts_dict[event_num] = event_date
    
    
    dataset = pd.read_csv("/Users/alanalrechah/Desktop/Uni/Thesis/concept-drift-driven-retraining-strategies-for-predictive-process-monitoring/Data/sww_90.csv")
    old_index = 0
    sublogs = []
    for key, value in drifts_dict.items():
        sub = dataset[old_index:key]
        sublogs.append(sub)
        old_index = key+1

    #append list sublog
    sub = dataset[old_index:len(dataset)]
    sublogs.append(sub)

    all_sublogs:list[SubLog] = []
    for sublog in sublogs:
        sublog = sublog.rename(columns={'case':'case:concept:name', 'role':'org:resource', 'event':'concept:name', 'completeTime':'time:timestamp'})
        print(sublog.columns)
        xes_log = pm4py.convert_to_event_log(sublog)
        log = EventLog(xes_log)
        result_sublog = SubLog(from_case=0, to_case=len(log._list), sub_log=log, event_log=sublog)
        all_sublogs.append(result_sublog)

    print(len(all_sublogs))
    return all_sublogs



