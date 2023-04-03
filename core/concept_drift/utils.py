import os
from pm4py.objects.log.obj import EventLog
from config import root_directory
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
import pandas as pd

from core.constants import ACTIVITY_IDENTIFIER_CSV, ACTIVITY_IDENTIFIER_XES, CASE_IDENTIFIER_CSV, CASE_IDENTIFIER_XES, RESOURCE_IDENTIFIER_CSV, RESOURCE_IDENTIFIER_XES, TIMESTAMP_IDENTIRIFIER_CSV, TIMESTAMP_IDENTIRIFIER_XES



def export_sampled_log(log: pd.DataFrame):
    """
    Exports the sampled log in both xes and csv formats
    Params - log: the event log needed to be exported
    """
    #export as csv
    log.to_csv(os.path.join(root_directory, "data", "output", "sampled.csv"))

    #export as xes
    renamed_log = log.rename(columns={CASE_IDENTIFIER_CSV: CASE_IDENTIFIER_XES,
                        ACTIVITY_IDENTIFIER_CSV: ACTIVITY_IDENTIFIER_XES,
                        TIMESTAMP_IDENTIRIFIER_CSV: TIMESTAMP_IDENTIRIFIER_XES,
                        RESOURCE_IDENTIFIER_CSV: RESOURCE_IDENTIFIER_XES})
    
    log_xes = pm4py.convert_to_event_log(renamed_log)
    xes_exporter.apply(log_xes, os.path.join(root_directory, "data", "output", "sampled.xes"))



def export_csv_as_xes(path: str, log_name: str, columns: dict[str, str]) -> EventLog:
    """
    Converts a csv dataframe to xes and export it
    Params - path: the path to the csv file
             log_name: is the name of the log that should be exported
             columns: a dict that specifies the names of the important columns
                        the keys should be: case, event, time, role
    
    Returns - the converted event log
    """
    log = pd.read_csv(path)

    log.rename(columns={columns["case"]: CASE_IDENTIFIER_XES, columns["event"]: ACTIVITY_IDENTIFIER_XES,
                        columns["time"]: TIMESTAMP_IDENTIRIFIER_XES, columns["role"]: RESOURCE_IDENTIFIER_XES}, inplace=True)
    
    log_xes = pm4py.convert_to_event_log(log)
    xes_exporter.apply(log_xes, os.path.join(root_directory, "data", "input", log_name + ".xes"))

    return log_xes


def export_xes_as_csv(path: str, log_name: str) -> pd.DataFrame:
    """
    Converts a xes eventlog to csv and export it
    Params - path: the path to the xes file
             log_name: is the name of the log that should be exported
    
    Returns - the converted event log 
    """
    log_xes = xes_importer.apply(path)

    log = pm4py.convert_to_dataframe(log_xes)

    log.rename(columns={CASE_IDENTIFIER_XES: CASE_IDENTIFIER_CSV,
                        ACTIVITY_IDENTIFIER_XES: ACTIVITY_IDENTIFIER_CSV,
                        TIMESTAMP_IDENTIRIFIER_XES: TIMESTAMP_IDENTIRIFIER_CSV,
                        RESOURCE_IDENTIFIER_XES: RESOURCE_IDENTIFIER_CSV})
    
    log.to_csv(os.path.join(root_directory, "data", "input", log_name + ".csv"))

    return log
    

