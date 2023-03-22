from pm4py.objects.log.obj import EventLog
from config import root_directory
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py



def export_sampled_log(log: EventLog):
    """
    Exports the sampled log in both xes and csv formats
    """
    xes_exporter.apply(log, root_directory + "data/output/sampled.xes")
    exported_df = pm4py.convert_to_dataframe(log)
    exported_df.rename(columns={'concept:name': 'event', 'org:resource': 'role',
                       'case:concept:name': 'case', 'time:timestamp': 'completeTime'}, inplace=True)
    exported_df.to_csv(root_directory + "CP/data/output/sampled.csv")