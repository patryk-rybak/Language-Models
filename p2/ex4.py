# DO POPRAWIENIA
# - ĠjÄĻzy

import torch
import re

from torch.nn import functional as F
from transformers import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'flax-community/papuGaPT2'
# model_name = 'eryk-mazus/polka-1.1b' # Context size: 2,048 tokens.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

max_length = 50
top_k = 30
top_p = 0.9
temperature = 1.5
specification = "Dokończ zdanie używając wyrazów zaczynających się na te same litery co prefiks. Prefiks: "
max_iterations = 50
ending_tokens = [tokenizer.convert_tokens_to_ids(token) for token in [".", "?", "!"]]

def constraint(token, allowed_letter):
    for i in 'Ċ:':
        if i in token: return False

    if token in {".", "?", "!"}: return True
    for i in '.,:;()[]{}+=-\\\/\|<>"':
        if i in token: return False
    if token.startswith(f"Ġ{allowed_letter.lower()}"): return True
    if not token.startswith(f"Ġ"): return True
    return False

def filter_tokens_by_letter(logits, allowed_letter):
    # zwrca scory z -inf dla zlych tokenow
    vocab = tokenizer.get_vocab()
    allowed_tokens = [token for token, index in vocab.items() if constraint(token, allowed_letter)]
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens]
    mask = torch.full(logits.shape, float("-inf")).to(device)
    mask[0, allowed_token_ids] = logits[0, allowed_token_ids]
    return mask

def apply_top_p(logits, top_p):
    cumulative_probs = torch.cumsum(F.softmax(logits, dim=-1), dim=-1)
    last_logit = (cumulative_probs > top_p).nonzero(as_tuple=True)[0][0].item()
    if last_logit == 0: last_logit = 1
    return logits[:last_logit]

def choose_best_candidate(candidates):
    scores = []
    for candidate in candidates:
        scores.append(sum([len(word)**2 for word in candidate.split()]))
        if candidate[-1] in {'.', '?', '!'}: scores[-1] += 500
    return candidates[scores.index(max(scores))]

with open("prefiksy.txt", "r") as f:
    prefixes = f.readlines()
    # prefixes amount
    for i in range(2):
        print()
        print('nowy prefix')
        prefix = prefixes[-i].strip()
        allowed_letter = prefix[0].lower()
        
        candidates = []
        with torch.no_grad():
            
            # number of candidates
            for i in range(3):
                print(f'Generating {i+1} candidate...')
                input_ids = tokenizer(specification + prefix, return_tensors="pt").input_ids.to(device)

                # candidate generation
                counter = 0
                while True:
                # for _ in range(max_iterations):
                    logits = model(input_ids).logits[:, -1, :]
                    filtered_logits = filter_tokens_by_letter(logits, allowed_letter)

                    if counter < 10:
                        for token in ending_tokens:
                            filtered_logits[:, token] -= 1000.0  # decrease ppb
                    elif counter > 15:
                        for token in ending_tokens:
                            filtered_logits[:, token] += 100.0  # increase pppb

                    # temperatura
                    filtered_logits = filtered_logits / temperature

                    # top-k
                    top_k_logits, top_k_indices = torch.topk(filtered_logits, top_k)

                    # top-p
                    top_p_logits = apply_top_p(top_k_logits, top_p)
                    top_p_indices = top_k_indices[:top_p_logits.shape[0]]
                    
                    # making propabilities
                    probs = torch.nn.functional.softmax(top_p_logits, dim=-1)

                    # sampling
                    sampled_token_index = torch.multinomial(probs, num_samples=1)
                    next_token_id = top_p_indices[0, sampled_token_index]

                    # updating history
                    # # generated_tokens.insert(0, next_token_id.item()) 
                    
                    # updating 
                    input_ids = torch.cat((input_ids, next_token_id), dim=1)
                    
                    # looking at 4 latest tokens to see if sentance is ended
                    decoded = tokenizer.decode(input_ids[0][-4::], skip_special_tokens=True)
                    if '.' in decoded or "?" in decoded or "!" in decoded:
                        print("BREAK")
                        break
                    
                    counter += 1

                
                # decoded_text = tokenizer.decode(list(reversed(generated_tokens)), skip_special_tokens=True)
                decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                decoded_text = decoded_text.replace(specification, '')
                candidates.append(decoded_text)
        
        # choosign best generations
        generated_text = choose_best_candidate(candidates)
        print('\ngenerated:\n' ,generated_text)
        print()
