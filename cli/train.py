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
        type=float,
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
        default=0.00,
        help=("How fast will your learning rate degrade"),
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help=("How long are your sequences"),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
    parser.add_argument(
        "--target_batch_size",
	type=int,
        default=32,
	help={"this can't run above bastch size 6 so this applies gradient accumulation"}
    )
    args = parser.parse_args()
    return args

def evaluate_model(model, device, dataloader, args):
    for batch in tqdm(dataloader, desc="Evaluating"):
      with torch.no_grad():
        model.eval()
        input_ids_v, att_mask_v, labels_v = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        if args.custom is False:
           valid_probs = model(input_ids=torch.squeeze(input_ids_v,1), attention_mask=att_mask_v, labels=torch.squeeze(labels_v,1)).logits
        else:
           valid_probs = model(input_ids=torch.squeeze(input_ids_v,1), attention_mask=att_mask_v, labels=torch.squeeze(labels_v,1))

        bleu.add_batch(predictions=torch.argmax(valid_probs, dim=-1), references=labels_v)

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

    if args.custom is False:
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_base_model"))
    elif args.custom is True:
        pre_trained = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_pretrained_model"))
        model = custom_mt5(mt5_model = pre_trained, seq_len = args.seq_len)
    
    model = model.to(device)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_tokenizer"), use_fast = True)
   # Datasets loaded from local files
    train_dataset = datasets.load_from_disk(os.path.join(args.save_dir, f"train_dataset"))
    eval_dataset = datasets.load_from_disk(os.path.join(args.save_dir, f"test_dataset"))
    # Dataloader
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors = 'pt')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collator, batch_size=args.batch_size, num_workers = 8)
    valid_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, collate_fn=collator, batch_size=args.batch_size, num_workers = 8)
    # accumative iterator for modulus
    accum_iter  = args.target_batch_size / args.batch_size
    # optimizer
    args.weight_decay = args.weight_decay*accum_iter
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # getting the max_train_steps and epoch count settled
    num_update_steps_per_epoch = len(train_dataloader)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # scheduler
    args.warmup_steps = args.warmup_steps / accum_iter
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


    # Time to train
    wandb.watch(model)
    global_step = 0
    epoch_bar = tqdm(range(args.num_epochs))
    progress_bar = tqdm(range(args.max_train_steps))
    for epoch in range(0, args.num_epochs):
        for batch_idx, example in enumerate(train_dataloader):
            input_ids, att_mask, labels = example["input_ids"].to(device), example["attention_mask"].to(device), example["labels"].to(device)
            with torch.set_grad_enabled(True):    
                if args.custom is False:
                   out = model(input_ids=torch.squeeze(input_ids,1), attention_mask=att_mask, labels=torch.squeeze(labels,1))
                   loss = out.loss
                   del out
                else:
                   logits = model(input_ids=torch.squeeze(input_ids,1), attention_mask=att_mask, labels=torch.squeeze(labels,1))
                   loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

                loss = loss / accum_iter
                loss.backward()

                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if(global_step == args.max_train_steps) or (global_step % args.eval_every == 0):
              results = evaluate_model(model, device, valid_dataloader, args)
              wandb.log({"eval/bleu": results["bleu"]},step=global_step)

            wandb.log({"train_loss": loss,
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "epoch": epoch}, step=global_step)

        if args.custom is False:
            torch.save(model, os.path.join(args.save_dir, f"{args.checkpoint_name}_base_model_trained.pt"))
        else:
            torch.save(model, os.path.join(args.save_dir, f"{args.checkpoint_name}_custom_model_trained.pt"))

        if global_step > args.max_train_steps:
            break
        epoch_bar.update(1)

    run.finish()  # stop wandb run

if __name__ == "__main__" :
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    main()
