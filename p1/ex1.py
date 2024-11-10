from transformers import pipeline
from transformers import logging

logging.set_verbosity_error()

history = []
messages_counter = 0
conversation_mem = 4
zero_input = '''Czym się zajmujesz?'''
assistant_start = 'Asystent: '
user_start = 'Użytkownik: '
specification_prompt = f'''Od teraz jesteś asystentem kulinarnym. Odpowiadasz na pytania użytkownika krótko i treściwie. Odpowiedzi rozpoczynasz w nowej linii od napisu "{assistant_start}" i kończ znakiem ".". Tylko raz używasz napisu "{assistant_start}". Nie wypisujesz napisu "{user_start}". Używasz następującego formatu:

Użytkownik: [Pytanie użytkownika]
Asystent: [Twoja odpowiedź]

Oto przykładowe odpowiedzi na pytania użytkownika:

Przykład 1:

{user_start}Jak zrobić omlet?
{assistant_start}Ubij jajka, dodaj sól, pieprz, smaż na maśle na małym ogniu przez 3-4 minuty.

Przykład 2:

{user_start}Jak upiec ciasto czekoladowe?
{assistant_start}Wymieszaj mąkę, cukier, kakao, jajka i masło. Piecz w 180°C przez 25-30 minut.

Przykład 3:

{user_start}Co dodać do sałatki?
{assistant_start}Dodaj świeże warzywa, oliwę z oliwek, ocet balsamiczny i przyprawy do smaku.

Pamiętaj, że jesteś asystentem i odpowiadasz na kulinarne pytania, nic więcej. Zaczynamy diaglog.

'''

def choose_best_response(responses):
    scores = []
    for i in range(len(responses)):
        score = 0
        proper_start = responses[i].startswith(assistant_start)
        user_start_counter = responses[i].count(user_start)
        assistant_start_counter = responses[i].count(assistant_start)
        length = len(responses)

        if proper_start: score += 100

        if assistant_start_counter == 1: score += 50
        else: score = -50

        if user_start_counter > 0: score -= 200

        score -= length
        scores.append(score)

    return scores.index(max(scores))

# generator = pipeline('text-generation', model='flax-community/papuGaPT2', device='cpu')
generator = pipeline("text-generation", model="eryk-mazus/polka-1.1b", device='cpu')
print('\nModel loaded\n')

history.append(specification_prompt)

while True:
    user_input = input('\nUżytkownik: ').strip()
    if not user_input:
        user_input = zero_input

    history.append(user_start + user_input)
    messages_counter += 1

    prompt = history[-min(messages_counter + 1 , conversation_mem):]
    # prompt = '\n'.join(prompt) + '\n' + assistant_start
    prompt = '\n'.join(prompt)
    prompt_len = len(prompt)

    responses = []
    for i in range(2):
        g = generator(prompt,
                    max_new_tokens=100,
                    pad_token_id=generator.tokenizer.eos_token_id,
                    temperature=0.9,
                    do_sample=True
                    )[0]['generated_text']
        responses.append(g[prompt_len:])

    best_response = responses[choose_best_response(responses)]

    history.append(assistant_start + best_response)
    messages_counter += 1
    print(best_response)