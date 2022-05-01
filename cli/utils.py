from copy import deepcopy
import random
import torch

def postprocess_text(preds, labels):
    """Use this function to postprocess generations and labels before BLEU computation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
