import pandas as pd
import datasets 
from datasets import load_dataset, concatenate_datasets

# source_tokenizer.save_pretrained(os.path.join(args.save_dir, f"{args.target_lang}_tokenizer"))

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
    encoded_dataset = dataset.map(preprocess_function, fn_kwargs = dict(seq_len = seq_len), remove_columns = old_col_names, num_proc = 4)
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
