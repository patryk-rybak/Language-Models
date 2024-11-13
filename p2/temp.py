import torch
import re
from transformers import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.set_verbosity_error()

model_name = 'flax-community/papuGaPT2'
# model_name = 'eryk-mazus/polka-1.1b' # Context size: 2,048 tokens.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)