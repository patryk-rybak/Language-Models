from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='flax-community/papuGaPT2', device='cpu')
# generator = pipeline("text-generation", model="eryk-mazus/polka-1.1b", device='cpu')

print ('Model loaded')
prompts = [
    # 'Dzisiaj na obiad zjemy kartofelki z',
    'Co słychać?',
    'Co słychać? '
]
for prompt in prompts:
    g = generator(prompt, 
       pad_token_id=generator.tokenizer.eos_token_id,
       max_new_tokens=50)[0]['generated_text']
    print (g)
    print (50 * '=')
    print ()
    
    
    
    
