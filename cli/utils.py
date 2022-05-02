from copy import deepcopy
import random
import torch

def postprocess_text(preds, labels):
    """Use this function to postprocess generations and labels before BLEU computation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    for i in range(0,len(preds)):
        if labels[i] is None or labels[i] is "":
           preds.pop(i)
           labels.pop(i)

    return preds, labels
