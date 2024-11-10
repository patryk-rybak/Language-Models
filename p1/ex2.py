import torch
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from transformers import logging

logging.set_verbosity_error()


# model_name = 'flax-community/papuGaPT2'
model_name = 'eryk-mazus/polka-1.1b'
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

def reduce_words_by_pairs(words):
    new_words = []
    while len(words) > 1:
        pairs = [sentence_prob(words[0] + ' ' + w)
                 for w in words[1:]] + [sentence_prob(w + ' ' + words[0])
                                        for w in words[1:]]
        
        max_ppb = max(pairs)
        idx = pairs.index(max_ppb)
        word_idx = (idx % (len(words) - 1)) + 1
        new_words.append(words[0] + ' ' + words[word_idx] if idx > len(pairs) // 2
                         else words[word_idx] + ' ' + words[0])
        words.pop(word_idx)
        words.pop(0)
    return new_words + words

def find_permutations(words):
    pbbs = []
    permutations = list(itertools.permutations(words))   
    for perm in permutations:
        sentence = ' '.join(perm).capitalize() + '.'
        pbbs.append(sentence_prob(sentence))
    sorted_perms_pbbs = sorted(zip(permutations, pbbs), key=lambda x: -x[1])
    return sorted_perms_pbbs # [(perm1, ppb1), ...]
    
def best_n_sentacne_permutations(sentence_text, n=2, threshold=4):
    assert n <= threshold
    words = sentence_text.lower()[:-1].split()

    while len(words) > threshold:
        words = reduce_words_by_pairs(words)
    
    return find_permutations(words)[:n]

sentences = [
    'To jest zwykłe polskie zdanie.',
    'This is a normal English sentence.',
    'Kasia kupiła trzy kilogramy jabłek.',
    'Hej, na imię mam Patryk.',
    'Przedstawił mi wszystkie możliwe opcje w związku z moim leczeniem.'
]

for s in sentences:
    for perm, ppb in best_n_sentacne_permutations(s):
        print(' '.join(perm).capitalize() + '.', ppb)
    print()