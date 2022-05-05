# NLP_FinalProject_2022
This is my last project for my Natural Language Processing class. In this project i have made my first  ever attempt at understanding and using a multilingual model as well as finding some way to modify or change it whether beneficially or interestingly. 

create_everything.py is meant to create the datasets that i would use as well as any downloaded and or modified versions of tokenizers or models I COULD or DID use <br>  
custom_mt had my weird extension of the mT5 model <br>  
utils holds my post process function which changes all the outputs and labels into a list each and then pops the corresponding indexed entries in both if the label list's entries came to be None <br>  
train held my train loop and my evaluation function

these were the args I used and next to them are the purpose of each: <br>  
--checkpoint_name = "Which checkpoint of mt5 do you want to use" / default was google/mt5-base
--save_dir        = "Directory which will be used to save everything. / no default / was required
--num_epochs      = "How many epochs" / defaulted to 1
--lr              = "learning rate i.e. how fast do you want it to learn / defaulted to 2e-5
--batch_size      = "How big is your batch" / defaulted to 8
--weight_decay    = "how fast will your learning rate decay" / defaulted to 0.00
--seq_len         = "how long are your sequences" / defaulted to 128,
--max_train_steps = "How how many steps will you train for" / no default / was not required
--eval_every      = "when will you evaluate your training / defaulted to 15k
--warmup_steps    = "when will you begin decaying your learning rate" / no default / was not required
--custom"         = "Am I training the custom extension model?" / defaulted to False,
--target_batch_size = "this can't run above bastch size 6 so this is the target for gradient accumulation" / defaulted to 32

In the end, I could not get the base model to work correctly for me, but regardless this is the function I used while attempting to get mT5 in working order.

regardless here is the functions I used: <br> 

python cli/create_everything.py --save_dir=output_dir

python3 cli/train.py --warmup_steps=12000 --save_dir=output_dir --batch_size=6 --target_batch_size=60 --eval_every=36000 --lr=1e-6 --max_train_steps=36000

