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

def constraint(token, allowed_letter):
    res = False
    if token.startswith(f"▁{allowed_letter.lower()}"): res = True
    elif token.startswith(f"▁{allowed_letter.upper()}"): res = True
    elif token.startswith(f",▁{allowed_letter.lower()}"): res = True
    elif token.startswith(f",▁{allowed_letter.upper()}"): res = True
    elif token in {".", "?", "!"}: res = True
    return res

def filter_tokens_by_letter(logits, allowed_letter):
    vocab = tokenizer.get_vocab()
    allowed_tokens = [token for token, index in vocab.items() if constraint(token, allowed_letter)]
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens]
    mask = torch.full(logits.shape, float("-inf"))
    mask[0, allowed_token_ids] = logits[0, allowed_token_ids]
    return mask

def apply_top_p(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    print("ppb")
    print(torch.nn.functional.softmax(sorted_logits, dim=-1))
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_logits = sorted_logits.masked_fill(cumulative_probs > top_p, float('-inf'))
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

with open("/home/patryk/Downloads/prefiksy.txt", "r") as f:
    prefixes = f.readlines()
    for i in range(1): # liczba prefixow
        prefix = prefixes[-i].strip()
        allowed_letter = prefix[0].lower()
        
        candidates = []
        for i in range(3): # liczba generacji
            input_ids = tokenizer(specification + prefix, return_tensors="pt").input_ids
            generated_tokens = [] 
            while True:
                logits = model(input_ids).logits[:, -1, :]
                filtered_logits = filter_tokens_by_letter(logits, allowed_letter)

                for j, token_id in enumerate(generated_tokens):
                    # Penalizacja zmniejsza się dla starszych tokenów w historii
                    decay_factor = penalty_weight - (penalty_decay * j)
                    decay_factor = max(decay_factor, 0.5)  # Minimalna penalizacja
                    filtered_logits[0, token_id] *= decay_factor

                filtered_logits = filtered_logits / temperature
                top_k_logits, top_k_indices = torch.topk(filtered_logits, top_k)
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # top_p_logits, top_p_indices = apply_top_p(top_k_logits, top_p)

                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                sampled_token_index = torch.multinomial(probs, num_samples=1)
                next_token_id = top_k_indices[0, sampled_token_index]

                generated_tokens.insert(0, next_token_id.item()) 
                
                input_ids = torch.cat((input_ids, next_token_id), dim=1)
                
                decoded = tokenizer.decode(input_ids[0][-4::], skip_special_tokens=True)
                print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                if re.search(r"[.] [A-ZĄĆĘŁŃÓŚŹŻ]", decoded) or "?" in decoded or "!" in decoded:
                    print("BREAK")
                    break
            
            decoded_text = tokenizer.decode(list(reversed(generated_tokens)), skip_special_tokens=True)
            candidates.append(prefix + decoded_text)
        
        generated_text = choose_best_candidate(candidates)
        print(generated_text)
        print()
