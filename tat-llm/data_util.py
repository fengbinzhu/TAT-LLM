import numpy as np
from tatqa_utils import to_number, is_number
import re

DATASET_FINQA = 'finqa'
DATASET_TATQA = 'tatqa'
DATASET_TATDQA='tatdqa'

HEAD_CLASSES_ = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "ARITHMETIC": 4}

OPERATOR = ['+', '-', '*', '/', '**', '>']

SCALE = ["", "thousand", "million", "billion", "percent"]


def get_head_class(answer_type: str, answer_mapping: dict, HEAD_CLASSES):
    Head_class = None
    if answer_type == "span":
        if len(answer_mapping) > 0:
            key = list(answer_mapping[0].keys())[0]
            if "paragraph" in key:
                Head_class = HEAD_CLASSES["SPAN-TEXT"]
            if "table" in key:
                Head_class = HEAD_CLASSES["SPAN-TABLE"]
    elif answer_type == "multi-span":
        Head_class = HEAD_CLASSES["MULTI_SPAN"]
    elif answer_type == "count":
        Head_class = HEAD_CLASSES["COUNT"]
    else:
        Head_class = HEAD_CLASSES["ARITHMETIC"]

    return Head_class

