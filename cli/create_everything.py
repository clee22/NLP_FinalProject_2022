# i intend this file to load everything (e.g. tokenizer, model, datasets, etc) and then save them locally

import os
import argparse
import logging
from packaging import version

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def parse_args():
    parser = argparse.ArgumentParser(description="create everything")

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="google/mt5-base",
        help=("Which checkpoint of mt5 do you want to you"),
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory which will be used to save everything.")

    args = parser.parse_args()
    return args
    
def main():
  args = parse_args()
  logger.info(f"Starting to make everything with args {args}")
  ######################################## LANG TOKEN MAP ########################################
  # since I dont plan on changing the datasets for this I will make the lang token map hardcoded 
  lang_tokens = {'en': '<en>', 'hu': '<hu>', 'ru': '<ru>', 'de': '<de>', 'es': '<es>'}
  ########################################## Tokenizer ###########################################
  logger.info(f"Loading pretrained tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_name, use_fast=True)
  logger.info(f"Adding language tokens to pretrained tokenizer")
  lang_tok_map = {"additional_special_tokens" : list(lang_tokens.values())}
  tokenizer.add_special_tokens(lang_tok_map)
  logger.info(f"Saving pretrained tokenizer")
  tokenizer.save_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_tokenizer"))
  ######################## Base Model For Custom Model & Comparison ##############################
  # the idea is i load it twice but save both the custom model and the comparison model
  # by doing so i intend so that i will not ever need to run this script again to get a fresh model
  # from re downloading the checkpoint
  logger.info(f"Loading model")
  base_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_name)
  pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_name,config=AutoConfig.from_pretrained(args.checkpoint_name,output_hidden_states=True)
  logger.info(f"Resizing model embeddings")
  base_model.resize_token_embeddings(len(tokenizer))
  pretrained_model.resize_token_embeddings(len(tokenizer))
  logger.info(f"Saving model")
  torch.save(base_model, os.path.join(args.save_dir, f"{args.checkpoint_name}_base_model.pt"))
  torch.save(pretrained_model, os.path.join(args.save_dir, f"{args.checkpoint_name}_pretrained_model.pt"))
  ##################################### Dataset setup ###########################################
  logger.info(f"calling set_up_datasets")
  train_set, eval_set = set_up_datasets(tokenizer, seq_len = 128):
  logger.info(f"saving combined datasets")
  train_set.save_to_disk(os.path.join(args.save_dir, f"train_dataset")
  eval_set.save_to_disk(os.path.join(args.save_dir, f"test_dataset"))
    
# for preprocessor function and dataset processing
def set_up_datasets(tokenizer, seq_len = 128):
  en_hu_dataset = load_dataset("tatoeba", lang1="en", lang2="hu") 
  de_hu_dataset = load_dataset("tatoeba", lang1="de", lang2="hu")
  de_es_dataset = load_dataset("tatoeba", lang1="de", lang2="es")
  es_ru_dataset = load_dataset("tatoeba", lang1="es", lang2="ru")
  # dataframe set up
  en_hu_pd = pd.DataFrame(data=en_hu_dataset['train']['translation'])
  de_hu_pd = pd.DataFrame(data=de_hu_dataset['train']['translation'])
  de_es_pd = pd.DataFrame(data=de_es_dataset['train']['translation'])
  es_ru_pd = pd.DataFrame(data=es_ru_dataset['train']['translation'])
  # front directional dataset setup
  en_hu = Dataset.from_pandas(en_hu_pd)
  de_hu = Dataset.from_pandas(de_hu_pd)
  de_es = Dataset.from_pandas(de_es_pd)
  es_ru = Dataset.from_pandas(es_ru_pd)
  # back directional dataset setup
  hu_en = Dataset.from_pandas(en_hu_pd[reversed(list(en_hu_pd.columns))])
  hu_de = Dataset.from_pandas(de_hu_pd[reversed(list(de_hu_pd.columns))])
  es_de = Dataset.from_pandas(de_es_pd[reversed(list(de_es_pd.columns))])
  ru_es = Dataset.from_pandas(es_ru_pd[reversed(list(es_ru_pd.columns))])
  # set up the tokens for the datasets
  lang_tokens = {'en': '<en>', 'hu': '<hu>', 'ru': '<ru>', 'de': '<de>', 'es': '<es>'}
  # create a token map of the languages and them to the token embeddings of the tokenizer
  lang_tok_map = {"additional_special_tokens" : list(lang_tokens.values())}
  tokenizer.add_special_tokens(lang_tok_map)
  model.resize_token_embeddings(len(tokenizer))
  # split each of the datasets into a 9:1 test split leaving approx 9000 test samples combined
  unprocessed_datasets = [en_hu, hu_de, de_es, es_ru, hu_en, de_hu, es_de, ru_es]
  split_datasets = []
  for dataset in unprocessed_datasets:
    split_datasets.append(dataset.train_test_split(test_size=0.1))
  # Now we map each one separately
  encoded_datasets = []
  for dataset in split_datasets:
    old_col_names = list(dataset['train'].column_names)
    encoded_dataset = dataset.map(preprocess_function, fn_kwargs = dict(seq_len = seq_len), remove_columns = old_col_names, batched = True, num_proc = 4)
    encoded_datasets.append(encoded_dataset)
  # now list the train splits and the test splits, and concatenate them
  train_splits = []
  test_splits = []
  for dataset in encoded_datasets:
    train_splits.append(dataset['train'])
    test_splits.append(dataset['test'])

  train_dataset = concatenate_datasets(train_splits)
  eval_dataset = concatenate_datasets(test_splits)
  
  return train_dataset, test_dataset

def preprocess_function(example, seq_len=128, Lang_token_map = lang_tokens):
  source, target = list(example.keys())
  task_prefix = Lang_token_map[target]
  encoding = tokenizer(
    task_prefix + example[source],
    padding="max_length",
    max_length=seq_len,
    truncation=True,
    return_tensors="pt",
  )
  input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

  target_encoding = tokenizer(example[target], padding="max_length", max_length=seq_len, truncation=True)
  labels = target_encoding.input_ids

  out_cols = ['input_ids', 'attn_mask', 'labels']
  out_items = [input_ids, attention_mask, labels]

  return dict(zip(out_cols, out_items))

if __name__ == "__main__" :
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    main()
