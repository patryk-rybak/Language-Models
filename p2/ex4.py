# DO POPRAWIENIA

import torch
import re
from transformers import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.set_verbosity_error()

# model_name = 'flax-community/papuGaPT2'
model_name = 'eryk-mazus/polka-1.1b' # Context size: 2,048 tokens.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

max_length = 50
top_k = 30
top_p = 0.9
temperature = 1.5
penalty_weight = 0.8
penalty_decay = 0.05
specification = "Dokończ zdanie używając wyrazów zaczynających się na te same litery co prefiks. Prefiks: "
max_iterations = 25
punctuation_tokens = [tokenizer.convert_tokens_to_ids(token) for token in [".", ",", "?", "!"]]

def constraint(token, allowed_letter):
    res = False
    if token.startswith(f"▁{allowed_letter.lower()}"): res = True
    elif token.startswith(f"▁{allowed_letter.upper()}"): res = True
    elif token.startswith(f",▁{allowed_letter.lower()}"): res = True
    elif token.startswith(f",▁{allowed_letter.upper()}"): res = True
    elif token in {".", "?", "!"}: res = True
    #elif not token.startswith(f"▁"): res = True
    return res

def filter_tokens_by_letter(logits, allowed_letter):
    vocab = tokenizer.get_vocab()
    allowed_tokens = [token for token, index in vocab.items() if constraint(token, allowed_letter)]
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens]
    mask = torch.full(logits.shape, float("-inf"))
    mask[0, allowed_token_ids] = logits[0, allowed_token_ids]
    return mask

# def apply_top_p(logits, top_p):
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
#     sorted_logits = sorted_logits.masked_fill(cumulative_probs > top_p, float('-inf'))
#     return sorted_logits, sorted_indices
def apply_top_p(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
    return sorted_logits, sorted_indices

def choose_best_candidate(candidates):
    cands = candidates.copy()
    scores = [0 for _ in range(len(candidates))]
    for i in range(len(candidates)):
        match = re.search(r"[.!?] ?[A-ZĄĆĘŁŃÓŚŹŻ]", candidates[i])
        if match:
            end_position = match.start() + 1
            cands[i] = candidates[i][:end_position]
        else:
            cands[i] = candidates[i][:max(candidates[i].find("?"), candidates[i].find("!")) + 1]

        score = len(cands[i]) * 20
        score += 50 if "," in cands[i] and ", " in cands[i] else 0
        score -= 50 if "," in cands[i] and ", " not in cands[i] else 0
        scores[i] = score
    print(cands)
    return cands[scores.index(max(scores))]

with open("prefiksy.txt", "r") as f:
    prefixes = f.readlines()
    # prefixes amount
    for i in range(1):
        print()
        print('nowy prefix')
        prefix = prefixes[-i].strip()
        allowed_letter = prefix[0].lower()
        
        candidates = []
        with torch.no_grad():

            # number of candidates
            for i in range(3):
                print(f'Generating {i+1} candidate...')
                input_ids = tokenizer(specification + prefix, return_tensors="pt").input_ids
                # # generated_tokens = [] 

                # candidate generation
                for _ in range(max_iterations):
                    logits = model(input_ids).logits[:, -1, :]
                    filtered_logits = filter_tokens_by_letter(logits, allowed_letter)

                    # w tym miejscu wypisz mi pozostałe tokeny
                    remaining_tokens = torch.topk(filtered_logits, k=filtered_logits.size(-1)).indices[0].tolist()
                    remaining_tokens_text = tokenizer.convert_ids_to_tokens(remaining_tokens)
                    print("Pozostałe tokeny po filtrowaniu:", remaining_tokens_text)

                    exit()

                    # Adjust probability of punctuation tokens
                    current_length = input_ids.shape[1]
                    if current_length < 20:
                        for token in punctuation_tokens:
                            filtered_logits[:, token] -= 100.0  # Decrease probability significantly
                    elif current_length > 30:
                        for token in punctuation_tokens:
                            filtered_logits[:, token] += 100.0  # Increase probability significantly

                    # temperatura
                    filtered_logits = filtered_logits / temperature

                    # top-k
                    top_k_logits, top_k_indices = torch.topk(filtered_logits, top_k)

                    # top-p
                    # # top_p_logits, top_p_indices = apply_top_p(top_k_logits, top_p)
                    # # print(top_p_indices)
                    # # print(top_p_logits)

                    # making propabilities
                    probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

                    # sampling
                    sampled_token_index = torch.multinomial(probs, num_samples=1)
                    next_token_id = top_k_indices[0, sampled_token_index]

                    # updating history
                    # # generated_tokens.insert(0, next_token_id.item()) 
                    
                    # updating 
                    input_ids = torch.cat((input_ids, next_token_id), dim=1)
                    
                    # looking at 4 latest tokens to see if sentance is ended
                    decoded = tokenizer.decode(input_ids[0][-4::], skip_special_tokens=True)
                    if re.search(r"[.] [A-ZĄĆĘŁŃÓŚŹŻ]", decoded) or "?" in decoded or "!" in decoded:
                        print("BREAK")
                        break
                
                # decoded_text = tokenizer.decode(list(reversed(generated_tokens)), skip_special_tokens=True)
                decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                decoded_text = decoded_text.replace(specification, '')
                candidates.append(decoded_text)
        
        # choosign best generations
        generated_text = choose_best_candidate(candidates)
        print('generated:' ,generated_text)
        print()
