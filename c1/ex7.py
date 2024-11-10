from transformers import pipeline
from transformers import logging

logging.set_verbosity_error()

specification_prompt_1 = '''Twoim zadaniem jest przetłumaczyć poniższy tekst z języka angielskiego na język polski, zachowując naturalność i poprawność gramatyczną w języku docelowym. Skup się na wiernym oddaniu sensu, stylu i tonu oryginalnego tekstu. Tłumaczenie pownno być wierne orginałowi.

Tekst w języku angielskim:'''
specification_prompt_2 = '\nTłumaczenie na język polski:'

generator = pipeline('text-generation', model='eryk-mazus/polka-1.1b', device='cpu')
print('Model loaded')

while True:
    user_input = input('\nUser:\n').strip()
    if user_input.strip() == '':
        print('\n')
        continue

    prompt = '\n'.join([specification_prompt_1, user_input, specification_prompt_2])
    prompt_len = len(prompt)

    g = generator(prompt,
                max_new_tokens=100,
                pad_token_id=generator.tokenizer.eos_token_id,
                )[0]['generated_text']
    response = g[prompt_len:].strip()

    print('\nTranslation:\n' + response)