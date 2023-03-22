import copy
from core.concept_drift.classes.sublog import SubLog



class PrioritySampling:

    def apply_priority_sampling(self, sublogs: list[SubLog], last: bool = True) -> list[SubLog]:
        """
        Apply priority sampling by giving weights to each sublog
        Params - sublogs: the list of sublogs
                 last: a boolean indicating if it should be priroty last or priroity first

        Returns - list of sublogs afer giving weights
        """
        # give an initial weight of 1
        for sublog in sublogs:
            sublog.weight = 1

        # get a copy of the sublogs and edit that copy
        remain_sublogs = copy.deepcopy(sublogs)

        if last:
            # apply the recursive priority last weighting function
            sublogs = self.apply_priority_weighting_last(sublogs, remain_sublogs)
        else:
            # apply the recursive priority last weighting function
            sublogs = self.apply_priority_weighting_first(sublogs, remain_sublogs, 0)

        return sublogs


    def apply_priority_weighting_last(self, all_sublogs: list = [], remaining_sublogs: list = []):
        """
        Apply the recursive priority last weighting technique
        Params - all_sublogs: all the sublogs
                 remaining_sublogs: list of the remained unweighted sublogs
        """
        # if no remaining sublogs then return the modified all sublogs
        if len(remaining_sublogs) == 0:
            return all_sublogs

        # iterate over sublogs and assign half the current weight
        for i, sublog in enumerate(remaining_sublogs):
            # check if last item then apply same as preceding item weight so it sums up to 1
            if len(remaining_sublogs) == 1:
                sublog.weight = all_sublogs[1].weight
                all_sublogs[i] = sublog
                continue

            sublog.weight = sublog.weight * 0.5
            all_sublogs[i] = sublog

        
        # delete the last element and then recurse
        del remaining_sublogs[-1]
        
        return self.apply_priority_weighting_last(all_sublogs, remaining_sublogs)
    

    def apply_priority_weighting_first(self, all_sublogs: list = [], remaining_sublogs: list = [], index: int = 0):
        """
        Apply the recursive priority first weighting technique
        Params - all_sublogs: all the sublogs
                 remaining_sublogs: list of the remained unweighted sublogs
                 index: to keep track of the sublog in all_sublogs not in remaining_sublog
        """
        # if no remaining sublogs then return the modified all sublogs
        if len(remaining_sublogs) == 0:
            return all_sublogs

        # iterate over sublogs and assign half the current weight
        for i, sublog in enumerate(remaining_sublogs):
            # check if last item then apply same as preceding item weight so it sums up to 1
            if len(remaining_sublogs) == 1:
                sublog.weight = all_sublogs[index - 1].weight
                all_sublogs[index + i] = sublog
                continue

            sublog.weight = sublog.weight * 0.5
            all_sublogs[i + index] = sublog

        # delete the first element then recurse
        del remaining_sublogs[0]
        
        return self.apply_priority_weighting_first(all_sublogs, remaining_sublogs, index+1)
    


class UniformSampling:

    def apply_uniform_sampling(self, sublogs: list[SubLog]) -> list[SubLog]:
        """
        Apply uniform sampling by giving weight to each sublog
        Params - sublogs: the list of sublogs
                 
        Returns - list of sublogs afer giving weights
        """
        # iterate over sublogs and give a uniform weight for each sublog
        for sublog in sublogs:
            sublog.weight = 1/len(sublogs)

        return sublogs
    

class OnlyLastSampling:

    def apply_only_last_sampling(self, sublogs: list[SubLog]) -> list[SubLog]:
        """
        Apply only last sublog sampling by giving weight to each sublog
        Params - sublogs: the list of sublogs
                 
        Returns - list of sublogs afer giving weights
        """
        # give only weight of 1 to the last sublog
        sublogs[len(sublogs) - 1].weight = 1

        return sublogs