import numpy as np
import pandas as pd

from core.constants import CASE_IDENTIFIER_CSV, TIMESTAMP_IDENTIRIFIER_CSV


class DecaySampling:

    def apply_exponential_decay(event_log: pd.DataFrame, decay_rate: float) -> pd.DataFrame:
        """
        Applies exponential decay on event log based on the given decay rate.

        Params:
            event_log: Input event log dataframe.
            decay_rate: Decay rate for exponential decay.

        Returns:
            pd.DataFrame: Event log dataframe with weights updated based on exponential decay.
        """

        # Convert timestamp to datetime object
        event_log[TIMESTAMP_IDENTIRIFIER_CSV] = pd.to_datetime(event_log[TIMESTAMP_IDENTIRIFIER_CSV])

        # Sort event log by timestamp
        event_log = event_log.sort_values(TIMESTAMP_IDENTIRIFIER_CSV)

        # Calculate the time differences in seconds for each case
        #max_timestamps = event_log.groupby(CASE_IDENTIFIER_CSV)[TIMESTAMP_IDENTIRIFIER_CSV].max()
        max_timestamp = event_log[TIMESTAMP_IDENTIRIFIER_CSV].max()

        event_log['time_diff'] = event_log.apply(lambda row: (max_timestamp - row[TIMESTAMP_IDENTIRIFIER_CSV]).total_seconds(), axis=1)
        time_diff = event_log.apply(lambda row: (max_timestamp - row[TIMESTAMP_IDENTIRIFIER_CSV]).total_seconds(), axis=1)
        time_dif_sum = event_log['time_diff'].sum()
        #normalize time difference
        event_log['time_diff'] = event_log['time_diff'] / time_dif_sum


        # Calculate the weights using the exponential decay function
        event_log['weight1'] = np.exp(-decay_rate * event_log['time_diff'])
        www = np.exp(-decay_rate * event_log['time_diff'])

        # Normalize the weights for each case
        #event_log["sum_weight"] = event_log.groupby(CASE_IDENTIFIER_CSV)["weight"].transform("sum")
        weights_sum = event_log["weight1"].sum()
        event_log['weight'] = event_log['weight1'] / weights_sum

        weights = event_log['weight'].values
        weights_sumsss = event_log['weight'].values.sum()

        #event_log['weight'] = event_log.groupby(CASE_IDENTIFIER_CSV)['weight'].apply(lambda x: x/x.sum())

        # Drop the time_diff column
        # event_log.drop('time_diff', axis=1, inplace=True)
        # event_log.drop('sum_weight', axis=1, inplace=True)

        result_log = event_log
        result_log[TIMESTAMP_IDENTIRIFIER_CSV] = result_log[TIMESTAMP_IDENTIRIFIER_CSV].astype(str)

        #result_log.to_excel("/Users/alanalrechah/Desktop/Uni/Thesis/Code/Concept-Drift-Retraining-Strategies-for-Predictive-Process-Monitoring/dd.xlsx", index=None, header=True)

        return result_log
    

    def sample_cases_original_log(event_log: pd.DataFrame, samples_num:int) -> pd.DataFrame:
        """
        Samples cases from original event log depending on cases weights
        
        Params:
            event_log: Input event log dataframe.
            samples_num: Number of cases to be sampled.

        Returns:
            Sampled Event log as Dataframe.
        """

        # Extract weights from event log
        weights = event_log['weight'].values

        indecies = event_log.index

        # Sample cases based on their weights
        sampled_indices = np.random.choice(indecies, samples_num, p=weights)
        
        # Extract sampled cases from event log
        sampled_cases = event_log.loc[sampled_indices][CASE_IDENTIFIER_CSV].values
        
        # Create new event log with sampled cases
        sampled_event_log = event_log[event_log[CASE_IDENTIFIER_CSV].isin(sampled_cases)]
        sampled_event_log.drop('weight', axis=1, inplace=True)

        return sampled_event_log
    


    def equalize_activities_num(sampled_log_path:str, test_log_path:str):
        """
        Equalize activities number on both train and test logs
        Params - sampled_log_path: the path of the sampled log
                 test_log_path: the path of the test log
        """

        sampled_log = pd.read_csv(sampled_log_path, index_col=0)
        test_log = pd.read_csv(test_log_path, index_col=0)

        train_act = sampled_log["event"].unique()
        test_act = test_log["event"].unique()

        print(len(train_act))
        print(len(test_act))
        dif = list(set(train_act) - set(test_act)) + list(set(test_act) - set(train_act))
        print("DIF: ", dif)
        #test = test.drop(columns=["Unnamed: 0"])

        if len(train_act) > len(test_act):
            for act in dif:
                last = test_log.tail(1)
                new_index = len(test_log)
                last["event"] = act
                new_data = pd.DataFrame(last.values, index=[new_index], columns=test_log.columns)
                test_log = pd.concat([test_log, new_data])
                print(test_log.tail(1))

            test_log.to_csv(test_log_path)
    
        elif len(train_act) < len(test_act):
            for act in dif:
                last = sampled_log.tail(1)
                new_index = len(sampled_log)
                last["event"] = act
                new_data = pd.DataFrame(last.values, index=[new_index], columns=sampled_log.columns)
                sampled_log = pd.concat([sampled_log, new_data])
                print(sampled_log.tail(1))

            sampled_log.to_csv(sampled_log_path)


        return sampled_log, test_log