import torch
import torch.nn as nn
import torch.nn.functional as F

class custom_mt5(nn.Module):
    def __init__(self, mt5_model, seq_len, dropout=0.0):
      super().__init__()
      self.model_layer = mt5_model
      self.addi_layers = nn.Sequential(
          nn.Linear(mt5_model.config.d_model, mt5_model.config.d_ff),
          nn.ReLU(),
          nn.Linear(mt5_model.config.d_ff, mt5_model.config.d_model)
      )
      self.output_layer = nn.Linear(mt5_model.config.d_model,mt5_model.config.vocab_size)
      self.layer_norm = nn.BatchNorm1d(seq_len)
      self.dropout = nn.Dropout(dropout) 

    def forward(self, input_ids, attention_mask, labels):
      encod = self.model_layer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      encod = self.layer_norm(encod.decoder_hidden_states[-1])
      encod = self.addi_layers(encod)
      out = self.output_layer(encod)
      return out

   def generate(input_ids, att_mask, tokenizer):
     generated_tokens = self.model_layer.generate(input_ids=input_ids,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            attention_mask=att_mask,
                                            do_sample=False,
                                            num_beams=5,
                                            )
     return generated tokens
