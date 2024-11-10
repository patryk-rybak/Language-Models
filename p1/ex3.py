import torch
import random
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from transformers import logging
from tqdm import tqdm
logging.set_verbosity_error()

model_name = 'flax-community/papuGaPT2'
# model_name = 'eryk-mazus/polka-1.1b'
device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label
            
def sentence_prob(sentence_txt):
    input_ids = tokenizer(sentence_txt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], input_ids[:, 1:])
        seq_log_probs = torch.sum(log_probs)
    return seq_log_probs.cpu().numpy()

total_lines = sum(1 for _ in open('reviews_for_task3.txt', 'r'))
with open('reviews_for_task3.txt', 'r') as reviews:
    score = 0
    lines_counter = 0
    for line in tqdm(reviews, total=total_lines):
        lines_counter += 1
        line = line.strip()
        if line.startswith('GOOD'):
            asnwer = 1
            line = line[5:]
        else:
            asnwer = 0
            line = line[4:]
        
        logppbGOOD = sentence_prob(line + '\nPolecam.')
        logppbBAD = sentence_prob(line + '\nNie polecam.')

        if logppbGOOD > logppbBAD:
            if asnwer == 1: score += 1
        elif asnwer == 0: score += 1

print(score / lines_counter)

# print('Neutralny teskst: Polecam.', sentence_prob('Neutralny teskst: Polecam.'))
# print('Neutralny teskst: Nie polecam.' ,sentence_prob('Neutralny teskst: Nie polecam.'))
# print('Polecam.', sentence_prob('Polecam.'))
# print('Nie polecam.', sentence_prob('Nie polecam.'))