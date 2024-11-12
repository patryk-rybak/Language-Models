import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from transformers import logging
from tqdm import tqdm
from math import inf
logging.set_verbosity_error()

model_name = 'eryk-mazus/polka-1.1b'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ZERO_SHOT
# specification_prompt = 'Odpowiedz na pytanie w sposób krótki i zwięzły. Odpowiedź powinna składać się z jednego słowa lub daty. Jeśli pytanie wymaga pojedynczego słowa, użyj najbardziej trafnego i odpowiedniego terminu.\nPytanie: '
specification_prompt = """
Twoim zadaniem jest udzielanie odpowiedzi na pytania w sposób faktyczny i zwięzły. 
Dla każdego pytania, udziel odpowiedzi w jednym z następujących formatów:
- Krótkie wyrażenie lub słowo.
- Liczba.
- Nazwa własna lub tytuł (np. imiona osób, miasta, wydarzenia).
- Jeśli pytanie dotyczy wyboru spośród kilku opcji, wybierz poprawną odpowiedź i podaj ją w odpowiedniej formie.


Przykłady:

Pytanie: W którym roku odbyła się bitwa pod Grunwaldem?  
Odpowiedź: 1410

Pytanie: Kto jest autorem powieści „Lalka”?  
Odpowiedź: Bolesław Prus

Pytanie: Jak nazywał się pierwszy człowiek na Księżycu?  
Odpowiedź: Neil Armstrong

Pytanie: Co oznacza łacińskie „suprema lex”?  
Odpowiedź: najwyższe prawo


Oto pytanie zasadnicze:

Pytanie: """
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
        candidates = list(map(lambda x: (x, sentence_prob(question + " " + x) if x else -inf), candidates))
        return candidates[candidates.index(max(candidates, key=lambda x: x[1]))][0]
    else:
        input_ids = tokenizer.encode(specification_prompt + question + '\nOdpowiedź:', return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_length=400, num_return_sequences=1)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.replace(specification_prompt + question + '\nOdpowiedź:', '').strip()

with open('task4_questions.txt', 'r') as questions, open('predicted_answers_ZERO_SHOT.txt', 'w') as answers:
# with open('czy_questions.txt', 'r') as questions, open('czy_prediciotns.txt', 'w') as answers:
    predicitons = []
    for question in tqdm(questions, total=sum(1 for _ in open('task4_questions.txt'))):
    # for question in tqdm(questions, total=sum(1 for _ in open('czy_questions.txt'))):
        question = question.strip()

        prediciton = find_asnwer(question)

        if prediciton == '': prediciotn = '2137'
        if len(prediciton.split('\n')) > 1: prediciton = prediciton.split('\n')[0]

        predicitons.append(prediciton)
    answers.write('\n'.join(predicitons))