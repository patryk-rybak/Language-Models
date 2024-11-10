import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from transformers import logging
from tqdm import tqdm
from math import inf
logging.set_verbosity_error()

model_name = 'eryk-mazus/polka-1.1b'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
specification_prompt = 'Bierzesz udział w turnieju z pytaniami. Twoim zadaniem jest poprawnie odpowiedzieć na pytanie. Odpowiedź musi byc krótka, najlepiej jednym słowem jeśli to możliwe. Powodzenia! Oto pytanie: '

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

def find_asnwer(question):
    # moze dodac jescze cos z "W którym" i "Kiedy" !!!!!!!!!!!
    if question.startswith('Czy') and ' czy ' not in question:
        return 'tak' if sentence_prob(question + ' Tak.') > sentence_prob(question + ' Nie.') else 'nie'
    elif question.startswith('Czy') and ' czy ' in question:
        words = question.split()
        czy_index = words.index('czy')
        candidates = [
            words[czy_index - 1],
            words[czy_index + 1],
            words[czy_index - 1] + ' ' + words[czy_index - 2] if czy_index - 2 >= 0 else None,
            words[czy_index + 1] + ' ' + words[czy_index + 2] if czy_index + 2 < len(words) else None
        ]
        candidates = list(map(lambda x: (x, sentence_prob(question + " " + x if x else -inf)), candidates))
        return candidates[candidates.index(max(candidates, key=lambda x: x[1]))][0]
    else:
        input_ids = tokenizer.encode(specification_prompt + question, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(input_ids, num_return_sequences=1)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.replace(specification_prompt, '').strip()
    reutrn -1
    
def is_valid(prediciton, answer):
    prediciton, answer = prediciton.lower(), answer.lower()
    # ???
    prediciton = prediciton.replace(',', '').replace('.', '')
    answer = answer.replace(',', '').replace('.', '')
    return prediciton == answer


with open('task4_questions.txt', 'r') as questions, open('task4_answers.txt', 'r') as answers:
    score = 0
    question_counter = 0
    for question, answer in tqdm(zip(questions, answers), total=sum(1 for _ in open('task4_questions.txt'))):
        question = question.strip()
        answer = answer.strip()

        prediciton = find_asnwer(question)
        if is_valid(prediciton, answer):
            score += 1
        question_counter += 1

print(score / question_counter)