import os
import argparse
import logging
from packaging import version

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.data.data_collator import DataCollatorWithPadding

# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()
# metric
bleu = datasets.load_metric("sacrebleu")

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


def main():
    args = parse_args()
    logger.info(f"Starting script with arguments: {args}")
    if args.custom is False:
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_base_model.pt")
    else:
        pre_trained =  AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_dir, f"{args.checkpoint_name}_pretrained_model.pt")
    
                                                             
        
        

    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(project=args.wandb_project, config=args)
    
    

if __name__ == "__main__" :
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    main()
