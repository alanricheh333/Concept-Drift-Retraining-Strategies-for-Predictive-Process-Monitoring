
from enum import Enum


CASE_IDENTIFIER_CSV = "case"
ACTIVITY_IDENTIFIER_CSV = "event"
TIMESTAMP_IDENTIRIFIER_CSV = "completeTime"
RESOURCE_IDENTIFIER_CSV = "role"



CASE_IDENTIFIER_XES = "case:concept:name"
ACTIVITY_IDENTIFIER_XES = "concept:name"
TIMESTAMP_IDENTIRIFIER_XES = "time:timestamp"
RESOURCE_IDENTIFIER_XES = "org:resource"




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