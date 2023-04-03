
import numpy as np
from core.concept_drift.classes.sublog import SubLog
from core.constants import CASE_IDENTIFIER_CSV


class OnlyFullCasesSamples:

    def sample_only_full_cases_from_buckets(self, sublogs: list[SubLog], sampled_buckets: list[int]) -> list:
        """
        Sample only full cases from buckets, this means it discards cases the span multiple buckets.
        Params - sublogs: list of sublogs after drift detection
                 sampled_buckets: the list of sampled buckets as indecies depending on a sampling method

        Retruns - list of sampled cases
        """
        
        list_of_sampled_cases = []

        #get unique cases for each bucket
        bucket_cases = {}
        for idx, sublog in enumerate(sublogs):
            bucket_cases[idx] = sublog.event_log[CASE_IDENTIFIER_CSV].unique()

        # sample a case from each time a certain bucket occures
        for bucket in sampled_buckets:
            #get the selected sublog
            sublog: SubLog = sublogs.__getitem__(bucket)

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

            #sample the whole case from the selected bucket(sublog)
            sampled_case = sublog.event_log.loc[sublog.event_log[CASE_IDENTIFIER_CSV] == sampled_case_id]

            list_of_sampled_cases.append(sampled_case)

        return list_of_sampled_cases
    


class CasesFromEventsSamples:

    def sample_cases_based_on_sampling_events(self, sublogs: list[SubLog], sampled_buckets: list[int]) -> list:
        """
        Sample cases from buckets depending on sampling events.
        Params - sublogs: list of sublogs after drift detection
                 sampled_buckets: the list of sampled buckets as indecies depending on a sampling method

        Retruns - list of sampled cases
        """

        list_of_sampled_cases = []

        #sample a case from each time a certain bucket occure
        for bucket in sampled_buckets:
            #get the selected sublog
            sublog: SubLog = sublogs.__getitem__(bucket)

            #sample an event from the selected bucket
            event = sublog.event_log.sample()
            case_id = event[CASE_IDENTIFIER_CSV].iloc[0]

            #sample the event corresponding case from the sublog
            sampled_case = sublog.event_log.loc[sublog.event_log[CASE_IDENTIFIER_CSV] == case_id]

            list_of_sampled_cases.append(sampled_case)

        return list_of_sampled_cases
    


class CasesBasedEventsCountSamples:

    def sample_cases_based_on_events_count(self, sublogs: list[SubLog], sampled_buckets: list[int]) -> list:
        """
        Sample cases depending on the higher count of events, if the cases spans more than one bucket
        then sample the case from the selected bucket only if it has the higher number of counted events.
        Params - sublogs: list of sublogs after drift detection
                 sampled_buckets: the list of sampled buckets as indecies depending on a sampling method

        Retruns - list of sampled cases
        """

        list_of_sampled_cases = []

        #get unique cases for each bucket
        bucket_cases = {}
        for idx, sublog in enumerate(sublogs):
            bucket_cases[idx] = sublog.event_log[CASE_IDENTIFIER_CSV].unique()

        #sample a case from each time a certain bucket occure
        for bucket in sampled_buckets:
            #get the selected sublog
            sublog: SubLog = sublogs.__getitem__(bucket)

            #sample a case id from the dedicated bucket
            sampled_case_id = np.random.choice(bucket_cases[bucket])

            #sample the case from the sublog
            sampled_case = sublog.event_log.loc[sublog.event_log[CASE_IDENTIFIER_CSV] == sampled_case_id]

            #get the count of events in the sampled case
            events_count = len(sampled_case)

            #boolean that identifies if there exists a bucket with a larger events count
            found_larger_count = False

            #iterate over the sublogs
            for index, item in enumerate(sublogs):
                #if the item(sublog) is not the current selected sublog
                if index != bucket:
                    
                    #get the events count of the sampled case id in the current item(sublog)
                    item_events_count = len(item.event_log.loc[item.event_log[CASE_IDENTIFIER_CSV] == sampled_case_id])
                    
                    #check if the item case events count is larger than the sampled sublog case events count
                    if item_events_count > events_count:
                        #give found True and break the loop
                        found_larger_count = True
                        break
            
            #if found is True then there exist another bucket with a higher count of events 
            #of the sampled case than the sampled bucket
            if found_larger_count:
                continue

            list_of_sampled_cases.append(sampled_case)

        return list_of_sampled_cases



class IncompleteCasesSamples:

    def sample_incomplete_cases(self, sublogs: list[SubLog], sampled_buckets: list[int]) -> list:
        """
        Sample cases from buckets event if incomplete and ignore that it spans other buckets.
        Params - sublogs: list of sublogs after drift detection
                 sampled_buckets: the list of sampled buckets as indecies depending on a sampling method

        Retruns - list of sampled cases
        """

        list_of_sampled_cases = []

        #get unique cases for each bucket
        bucket_cases = {}
        for idx, sublog in enumerate(sublogs):
            bucket_cases[idx] = sublog.event_log[CASE_IDENTIFIER_CSV].unique()

        #sample a case from each time a certain bucket occure
        for bucket in sampled_buckets:
            #get the selected sublog
            sublog: SubLog = sublogs.__getitem__(bucket)

            #sample a case id from the dedicated bucket
            sampled_case_id = np.random.choice(bucket_cases[bucket])

            #sample the case from the sublog
            sampled_case = sublog.event_log.loc[sublog.event_log[CASE_IDENTIFIER_CSV] == sampled_case_id]

            list_of_sampled_cases.append(sampled_case)

        return list_of_sampled_cases



class EventsSamples:

   def sample_only_events(self, sublogs: list[SubLog], sampled_buckets: list[int]) -> list:
        """
        Sample events from buckets and construct a sampled log from these events.
        Params - sublogs: list of sublogs after drift detection
                 sampled_buckets: the list of sampled buckets as indecies depending on a sampling method

        Retruns - list of sampled cases(events forming cases)
        """

        list_of_sampled_cases = []

        #sample a case from each time a certain bucket occure
        for bucket in sampled_buckets:
            #get the selected sublog
            sublog: SubLog = sublogs.__getitem__(bucket)

            #sample an event from the selected bucket
            sampled_case = sublog.event_log.sample()

            list_of_sampled_cases.append(sampled_case)

        return list_of_sampled_cases

