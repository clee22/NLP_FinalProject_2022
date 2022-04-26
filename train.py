############## imports ##############
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_metric

############## model & tokenizer ##############

model_name = "google/mt5-small"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)
############## metric ##############
bleu = load_metric("sacrebleu")
