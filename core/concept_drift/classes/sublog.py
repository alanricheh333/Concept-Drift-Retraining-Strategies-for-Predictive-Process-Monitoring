from pm4py.objects.log.obj import EventLog

class SubLog():
    from_case: int
    to_case: int
    sub_log: EventLog
    weight: float = 0.0
    sample_num: int
    event_log = None

    def __init__(self, from_case, to_case, sub_log, event_log=None):
        self.from_case = from_case
        self.to_case = to_case
        self.sub_log = sub_log
        self.event_log = event_log