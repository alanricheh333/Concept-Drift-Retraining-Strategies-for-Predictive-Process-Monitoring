from enum import Enum
from numpy import NaN
import Data
import Predictions.setting as pred
import Predictions.metric as met
import Methods
from Utils.LogFile import LogFile
from CP import concept_drift
import pm4py
from config import root_directory
import pandas as pd
import numpy as np
from subprocess import PIPE, Popen



def export_xes_to_csv(file_name):
    log = pm4py.read_xes(root_directory + "CP/data/input/script/" + file_name)
    ss = pm4py.convert_to_dataframe(log)
    ss.to_csv(root_directory + "Data/" + file_name)


def export_xes(new_name, old_name = None, data_frame = None, add_resource = False):

    df = data_frame
    #check if we need to get the name from the input folder
    if old_name:
        log = pm4py.read_xes(root_directory + "CP/data/input/script/"+ old_name + ".xes")
        #convert to dataframe
        df = pm4py.convert_to_dataframe(log)
    
    #add resource column with empty values
    if add_resource:
        for index, row in df.iterrows():
            temp = row["org:resource"]
            if row['org:resource'] is NaN:
                row["org:resource"] = ""
    
    #export to xes
    exported_log = pm4py.convert_to_event_log(df)
    pm4py.write_xes(exported_log, root_directory + "CP/data/input/script/" + new_name + ".xes")


def analyze():
    #train = LogFile(root_directory + "Data/BPIC18.csv", ",", 0, None, "startTime", "case", activity_attr=None, integer_input=False, convert=False)
    train = LogFile(root_directory + "Data/Helpdesk.csv", ",", 0, None, "completeTime", "case", "event", integer_input=False, convert=False)
    print("Columns: ", train.data.columns)
    print("Num of attributes:", len(train.data.columns))
    train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    print("Num of attributes:", len(train.data.columns))
    print(train.data.columns)

    for attr in train.data.columns:
        print(attr, len(train.data[attr].value_counts()))


def split_data(file_name, split_percent = 90, case_column = "case:concept:name", role_column = "org:resource", event_column="concept:name", time_column = "time:timestamp"):
    """
    split data in csv log with moving events in same case to train in case they are in test data
    """
    #import log
    log = pd.read_csv(root_directory + "Data/" + file_name + ".csv", header=0, nrows=None, delimiter=",", encoding='latin-1')
    
    #sort values by time
    #log = log.sort_values("completeTime")
    
    #split data into train and test
    train = log.head(int(len(log)*(split_percent/100)))

    #get unique cases in train set
    train_cases_ids = train[case_column].unique()

    #get the test data
    last_index = train.index[-1]
    test = log.loc[last_index:]
    test = test.iloc[1: , :]

    print("train: "+ str(len(train)))
    print("test: " + str(len(test)))

    #iterate over data and see if events from test data belong to cases in train data then move them
    for index, row in test.iterrows():
        if row[case_column] in train_cases_ids:
            train = train.append(row)
            test = test.drop(index)
            print("train: "+ str(len(train)))
            print("test: " + str(len(test)))

    #export train and test data to csv
    train.to_csv(root_directory + "Data/" + file_name + "_" + str(split_percent) + ".csv")
    test.to_csv(root_directory + "Data/" + file_name + "_"+ str(100-split_percent) +".csv")

    #export train data to xes
    train.rename(columns={'role': role_column, 'event':event_column, 'completeTime': time_column, 'case': case_column}, inplace=True)
    print(train.columns)
    del train["Unnamed: 0"]
    train["time:timestamp"] = pd.to_datetime(train["time:timestamp"])
    exported_train = pm4py.convert_to_event_log(train)
    pm4py.write_xes(exported_train, root_directory + "CP/data/input/script/"+ file_name + "_" + str(split_percent) + ".xes")


def naive_split_data(file_name, split_percent = 90, case_column = "case:concept:name", role_column = "org:resource", event_column="concept:name", time_column = "time:timestamp"):
    """
    split data naivly without moving events in same case in train and test
    """
    #import log
    log = pd.read_csv(root_directory + "Data/" + file_name + ".csv", header=0, nrows=None, delimiter=",", encoding='latin-1')
    
    #sort values by time
    #log = log.sort_values("completeTime")

    #rename columns for csv
    log.rename(columns={role_column: 'role', event_column: 'event', time_column: 'completeTime', case_column: 'case'}, inplace=True)

    if 'role' not in log.columns:
        log["role"] = ""
    
    #split data into train and test
    train = log.head(int(len(log)*(split_percent/100)))

    #get the test data
    last_index = train.index[-1]
    test = log.loc[last_index:]
    test = test.iloc[1: , :]

    #export train and test data to csv
    train.to_csv(root_directory + "Data/" + file_name + "_" + str(split_percent) + ".csv")
    test.to_csv(root_directory + "Data/" + file_name + "_"+ str(100-split_percent) +".csv")

    #export train data to xes
    train.rename(columns={'role': role_column, 'event':event_column, 'completeTime': time_column, 'case': case_column}, inplace=True)
    print(train.columns)
    
    if 'Unnamed: 0' in train.columns:
        del train["Unnamed: 0"]
    
    train["time:timestamp"] = pd.to_datetime(train["time:timestamp"])
    exported_train = pm4py.convert_to_event_log(train)
    pm4py.write_xes(exported_train, root_directory + "CP/data/input/script/"+ file_name + "_" + str(split_percent) + ".xes")

    

class TrainMethod(Enum):
    SDL = "sdl"
    LIN = "lin"
    PASQUADIBISCEGLIE = "pasquadibisceglte"
    


def predict_after_detection(method: TrainMethod = TrainMethod.SDL, train_file:str = "sample", test_file:str = "sww_test"):

    if method is TrainMethod.SDL:
        #start with new data
        train_data = Data.get_data(train_file)

        #select predection method
        settings = pred.SDL

        #select the percentage of training with is always 100%
        settings.train_percentage = 100
        #prepare data with predection method settings
        train_data.prepare(settings)

        #getting the prediction method parameters
        method = Methods.get_prediction_method("SDL")

        
        #get the test data
        test_data = Data.get_data(test_file)
        
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
        train_data = Data.get_data(train_file)

        #select predection method
        settings = pred.LIN

        #select the percentage of training with is always 100%
        settings.train_percentage = 100
        #prepare data with predection method settings
        train_data.prepare(settings)

        #getting the prediction method parameters
        method = Methods.get_prediction_method("LIN")

        
        #get the test data
        test_data = Data.get_data(test_file)
        
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
        train_data = Data.get_data(train_file)

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
        method = Methods.get_prediction_method("PASQUADIBISCEGLIE")

        
        #get the test data
        test_data = Data.get_data(test_file)

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
    
    #data_object = Data.get_data("IOR5k", time="time:timestamp", case="case:concept:name", activity="concept:name", resource="org:resource")
    #data_object = Data.get_data("sample")
    data_object = Data.get_data("sww_train")
    #data_object = Data.get_data("sample_sdl")

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

    #m = Methods.get_prediction_method("LIN")
    #m = Methods.get_prediction_method("SDL")
    m = Methods.get_prediction_method("PASQUADIBISCEGLIE")

    test_data = Data.get_data("sww_test")

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


def detect(customized_sampling: bool = False):
    if customized_sampling:
        #apply drift detection
        all_sublogs = concept_drift.apply_drift_detection(log_name="sww_90", delta=0.002)
        #apply general sampling
        result_log = concept_drift.general_sampling(all_sublogs, probabilites=[1, 0, 0, 0], case_percent=0.6, all_cases=False)
        # export result log
        concept_drift.export_sampled_log(result_log)
    else:
        concept_drift.detect_drifts(use_saved_sublogs=False, sampling_method=concept_drift.SamplingMethod.UNIFORM, log_name="sww_90", over_sampling=True, all_cases=False)



def apr():
    
    process = Popen(['java', '-jar', '/Users/alanalrechah/Desktop/Uni/Thesis/concept-drift-driven-retraining-strategies-for-predictive-process-monitoring/ProDrift2.5.jar', '-fp', '/Users/alanalrechah/Desktop/Uni/Thesis/concept-drift-driven-retraining-strategies-for-predictive-process-monitoring/Data/input/script/sww_90.xes',
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

    print(len(sublogs))

if __name__ == '__main__':
    #apr()
    #predict()
    #detect()
    predict_after_detection(method= TrainMethod.SDL)
    #naive_split_data(file_name="BPIC15_1_sorted_new")
