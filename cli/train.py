import os
import argparse
import logging
from packaging import version

from tqdm.auto import tqdm
import math
import wandb
import torch
import datasets
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, load_metric
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.data.data_collator import DataCollatorWithPadding
from custom_mt import custom_mt5 

"""
# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
"""

#datasets.utils.logging.set_verbosity_warning()
#transformers.utils.logging.set_verbosity_warning()
# metric
bleu = load_metric("sacrebleu")

def parse_args():
    parser = argparse.ArgumentParser(description="Every Param for training")

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="google/mt5-base",
        help=("Which checkpoint of mt5 do you want to use"),
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory which will be used to save everything.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=("How many epochs"),
    )
    parser.add_argument(
        "--lr",
        type=int,
        default=2e-5,
        help=("how much do you want it to learn"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=("How big is your batch"),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help=("How fast will your learning rate degrade"),
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help=("How long are your sequences"),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=35000,
        help=("How how many steps will you train for"),
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=15000,
        help=("when will you evaluate your training"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help=("when will you evaluate your training"),
    )
    parser.add_argument(
        "--custom",
        type=bool,
        default=False,
        help=("Am I training the custom or default model?"),
    )
    args = parser.parse_args()
    return args

def evaluate_model(model, device, dataloader):
    for batch in tqdm(dataloader, desc="Evaluating"):
      with torch.no_grad():
        model.eval()
        input_ids_v, att_mask_v, labels_v = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        valid_probs = model.forward(input_ids=torch.squeeze(input_ids_v,1), attention_mask=att_mask_v, labels=torch.squeeze(labels_v,1))

        bleu.add_batch(predictions=torch.argmax(valid_probs, dim=-1), references=labels_v)
    wandb.log({'bleu': bleu_value})
    model.train()
    eval_metric = bleu.compute()
    evaluation_results = {
        "bleu": eval_metric["score"],
    }
    return evaluation_results

def main():
    args = parse_args()
    # define device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # logger.info(f"Starting script with arguments: {args}")
    if args.custom is False:
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_base_model"))
    else:
        pre_trained =  AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_pretrained_model"))
        model = custom_mt5(mt5_model = pretrained_model)
    
    model = model.to(device)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_tokenizer"))
  # Datasets loaded from local files
    train_dataset = datasets.load_from_disk(os.path.join(args.save_dir, f"train_dataset"))
    eval_dataset = datasets.load_from_disk(os.path.join(args.save_dir, f"test_dataset"))
    # Dataloader
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors = "pt")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collator, batch_size=args.batch_size)
    valid_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, collate_fn=collator, batch_size=args.batch_size)    
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # getting the max_train_steps and epoch count settled
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.num_epochs = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # scheduler
    lr_scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Initialize wandb as soon as possible to log all stdout to the cloud
    run = wandb.init(project="Final ProjectSpring2022", config=args)
    
    #logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    #logger.info(f"  Num Epochs = {args.num_train_epochs}")
    #logger.info(f"  Total optimization steps = {args.max_train_steps}")
    #progress_bar = tqdm(range(args.max_train_steps))

    # Time to train
    global_step = 0
    for _ in tqdm(range(args.num_epochs), desc="Epochs"):
        for example in tqdm(train_dataloader, desc="Training"):
            model.train()
            input_ids, att_mask, labels = example["input_ids"].to(device), example["attention_mask"].to(device), example["labels"].to(device)
            if args.custom is False:
               out = model.forward(input_ids=torch.squeeze(input_ids,1), attention_mask=att_mask, labels=torch.squeeze(labels,1))
               loss = out.loss
            else:
               logits = model.forward(input_ids=torch.squeeze(input_ids,1), attention_mask=att_mask, labels=torch.squeeze(labels,1))
               loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if(global_step == argsmax_train_steps) or (global_step % eval_every == 0):
              results = evaluate_model(model, device, valid_dataloader)
              wandb.log({"eval/bleu": results["bleu"] }
                  
            if args.custom is False:
               torch.save(model, os.path.join(args.save_dir, f"{args.checkpoint_name}_base_model_trained.pt"))
            else:
               torch.save(model, os.path.join(args.save_dir, f"{args.checkpoint_name}_custom_model_trained.pt"))

    run.finish()  # stop wandb run

if __name__ == "__main__" :
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    main()
