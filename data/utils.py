import os
from typing import Tuple
import pm4py
import pandas as pd
import numpy as np
from config import root_directory
from pm4py.objects.log.obj import EventLog
from core.constants import ACTIVITY_IDENTIFIER_CSV, ACTIVITY_IDENTIFIER_XES, CASE_IDENTIFIER_CSV, CASE_IDENTIFIER_XES, RESOURCE_IDENTIFIER_CSV, RESOURCE_IDENTIFIER_XES, TIMESTAMP_IDENTIRIFIER_CSV, TIMESTAMP_IDENTIRIFIER_XES



def import_xes_eventlog(full_file_path: str, file_name: str, split_percent: int, columns_names: dict[str, str], naive: bool = True):
    """
    Imports xes event log to split and export it as train and test data for both csv and xes
    Params - full_file_path: the file full path
             file_name: the file name that exported files should have
             split_percet: the percentage of the split
             columns_names: a dict of the original columns names (control flow)
             naive: if the split should be naivly or not
    """

    #read the xes file
    xes_log = pm4py.read_xes(full_file_path)

    #convert to dataframe for the split
    converted_log = pm4py.convert_to_dataframe(xes_log)

    #split the data
    train_csv, test_csv, train_xes, test_xes = split_data(converted_log, columns_names, naive, split_percent)

    #export csv files
    train_csv.to_csv(os.path.join(root_directory, "data", "input", "csv", file_name + "_train.csv"))
    test_csv.to_csv(os.path.join(root_directory, "data", "input", "csv", file_name + "_test.csv"))

    #export xes files
    pm4py.write_xes(train_xes, os.path.join(root_directory, "data", "input", "xes", file_name + "_train.xes"))
    pm4py.write_xes(test_xes, os.path.join(root_directory, "data", "input", "xes", file_name + "_test.xes"))
    
    print("splitted and exported!")
    return True



def import_csv_eventlog(full_file_path: str, file_name: str, split_percent: int, columns_names: dict[str, str], naive: bool = True):
    """
    Imports csv event log to split and export it as train and test data for both csv and xes
    Params - full_file_path: the file full path
             file_name: the file name that exported files should have
             split_percet: the percentage of the split
             columns_names: a dict of the original columns names (control flow)
             naive: if the split should be naivly or not
    """

    #read the csv file
    csv_log = pd.read_csv(full_file_path, header=0, nrows=None, delimiter=",", encoding='latin-1')

    # csv_log["role"] = ""
    csv_log["lifecycle:transition"] = "complete"

    #split the data
    train_csv, test_csv, train_xes, test_xes = split_data(csv_log, columns_names, naive, split_percent)

    #export csv files
    train_csv.to_csv(os.path.join(root_directory, "data", "input", "csv", file_name + "_train.csv"))
    test_csv.to_csv(os.path.join(root_directory, "data", "input", "csv", file_name + "_test.csv"))

    #export xes files
    pm4py.write_xes(train_xes, os.path.join(root_directory, "data", "input", "xes", file_name + "_train.xes"))
    pm4py.write_xes(test_xes, os.path.join(root_directory, "data", "input", "xes", file_name + "_test.xes"))

    # if 'Unnamed: 0' in train.columns:
    #     del train["Unnamed: 0"]
    # train["time:timestamp"] = pd.to_datetime(train["time:timestamp"])
    
    print("splitted and exported!")
    return True

    

def split_data(log: pd.DataFrame, columns_names: dict[str, str], naive:bool =True, split_percent = 90) -> Tuple[pd.DataFrame, pd.DataFrame, EventLog, EventLog]:
    """
    split data in csv log with moving events in same case to train in case they are in test data
    Params - log: the event log as dataframe
             columns_names: a dict of the original columns names (control flow)
             naive: if the split should be naivly or not
             split_percent: the percentage of the split

    Returns - the splitted data (train, test) in both format csv and xes
    """
    
    #TODO: remove sort in the future if not needed
    #sort values by time
    log = log.sort_values(columns_names['timestamp'])
    log[columns_names['timestamp']] = pd.to_datetime(log[columns_names['timestamp']], format="%Y-%m-%d %H:%M:%S")#.dt.strftime("%Y-%m-%d %H:%M:%S")
    log['startTime'] = pd.to_datetime(log['startTime'], format="%Y-%m-%d %H:%M:%S")#.dt.strftime("%Y-%m-%d %H:%M:%S")
        

    #rename the columns
    try:
        log.rename(columns={columns_names['case']: CASE_IDENTIFIER_CSV,
                          columns_names["activity"]: ACTIVITY_IDENTIFIER_CSV,
                          columns_names['timestamp']: TIMESTAMP_IDENTIRIFIER_CSV,
                          columns_names["resource"]: RESOURCE_IDENTIFIER_CSV}, inplace=True)
    except:
        raise ValueError("columns_names should include case, activity, timestamp and resource. Please provide the values for those keys!")
    
    #split data into train and test
    train = log.head(int(len(log)*(split_percent/100)))

    #get the test data
    last_index = train.index[-1]
    test = log.loc[last_index:]
    test = test.iloc[1: , :]

    print("train: "+ str(len(train)))
    print("test: " + str(len(test)))   

    #if naive is false then split data with moving events belonging to cases in train set
    if naive is False:
        #get unique cases in train set
        train_cases_ids = train[CASE_IDENTIFIER_CSV].unique()

        #iterate over data and see if events from test data belong to cases in train data then move them
        for index, row in test.iterrows():
            if row[CASE_IDENTIFIER_CSV] in train_cases_ids:
                train = train.append(row)
                test = test.drop(index)
            
        print("train: "+ str(len(train)))
        print("test: " + str(len(test)))

    #convert train set to xes
    train_temp = train.rename(columns={CASE_IDENTIFIER_CSV: CASE_IDENTIFIER_XES,
                                ACTIVITY_IDENTIFIER_CSV: ACTIVITY_IDENTIFIER_XES,
                                TIMESTAMP_IDENTIRIFIER_CSV: TIMESTAMP_IDENTIRIFIER_XES,
                                RESOURCE_IDENTIFIER_CSV: RESOURCE_IDENTIFIER_XES})
    
    if train_temp.isnull().values.any():
        train_temp = train_temp.replace(np.nan, "", regex=True)
    
    train_xes = pm4py.convert_to_event_log(train_temp)

    #convert test set to xes
    test_temp = test.rename(columns={CASE_IDENTIFIER_CSV: CASE_IDENTIFIER_XES,
                                ACTIVITY_IDENTIFIER_CSV: ACTIVITY_IDENTIFIER_XES,
                                TIMESTAMP_IDENTIRIFIER_CSV: TIMESTAMP_IDENTIRIFIER_XES,
                                RESOURCE_IDENTIFIER_CSV: RESOURCE_IDENTIFIER_XES})
    
    if test_temp.isnull().values.any():
        test_temp = test_temp.replace(np.nan, "", regex=True)
    
    test_xes = pm4py.convert_to_event_log(test_temp)

    return train, test, train_xes, test_xes



def split_sampled_log(log: pd.DataFrame, naive:bool = True, split_percent = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the sampled event log into train and test logs
    Params - log: the event log as dataframe
             naive: if the split should be naivly or not
             split_percent: the percentage of the split

    Returns - the splitted data (train, test) in both format csv and xes
    """
    #split data into train and test
    train = log.head(int(len(log)*(split_percent/100)))

    #get the test data
    last_index = train.index[-1]
    test = log.loc[last_index:]
    test = test.iloc[1: , :]

    print("train: "+ str(len(train)))
    print("test: " + str(len(test)))   

    #if naive is false then split data with moving events belonging to cases in train set
    if naive is False:
        #get unique cases in train set
        train_cases_ids = train[CASE_IDENTIFIER_CSV].unique()

        #iterate over data and see if events from test data belong to cases in train data then move them
        for index, row in test.iterrows():
            if row[CASE_IDENTIFIER_CSV] in train_cases_ids:
                train = train.append(row)
                test = test.drop(index)
            
        print("train: "+ str(len(train)))
        print("test: " + str(len(test)))

    return train, test