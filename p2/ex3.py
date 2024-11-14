# POPRAWION

import torch
from transformers import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
logging.set_verbosity_error()

# model_name = 'flax-community/papuGaPT2'
model_name = 'eryk-mazus/polka-1.1b'
device = 'cpu'
# device = 'cuda'

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

raw_variants = "wprost|wyprosty|wyprostu|wyprost uwielbiała|wielbił|wielbiła|uwielbił|wielbiło|uwielbiał|uwielbiało|uwielbiały słuchać|osłuchać|słychać|usłuchać o|i|e|a|ó|ę|y|ą|u wartościach własnych|owłosionych macierzy|mocarz|macierzą|macierze|mocarza|mocarze|mocarzy|macierz"
variants = [ vs.split("|") for vs in raw_variants.split()]

class BeamSearch:
    def __init__(self, variantes, beam_width=3):
        self.variants = variants
        self.beam_width = beam_width
    
    def search(self):
        beams = [("", 0)]
        for i in range(len(self.variants)):
            new_beams = []
            for beam in beams:
                for variant in self.variants[i]:
                    new_sentence = variant if i == 0 else beam[0] + " " + variant
                    new_scoer = beam[1] + sentence_prob(new_sentence.capitalize() + "" if i != len(self.variants) - 1 else ".")
                    new_beam = (new_sentence, new_scoer)
                    new_beams.append(new_beam)
            beams = sorted(new_beams, key=lambda x: -x[1])[:self.beam_width]
            print(beams)
        return beams[0][0]

bs = BeamSearch(variants)
print(bs.search())
