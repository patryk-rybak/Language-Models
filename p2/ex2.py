# moze byc problem z generowaniem slow na wielkie litery
import torch
import pickle
from torch.nn import functional as F
from tqdm import tqdm
from transformers import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, tokens_ids):
        node = self.root
        for token_id in tokens_ids:
            if token_id.item() not in node.children:
                node.children[token_id.item()] = TrieNode()
            node = node.children[token_id.item()]
        node.is_word_end = True
            
    @classmethod
    def find_next_node(cls, starter_node, token_id):
        return starter_node.children.get(token_id.item(), None)

#znajduje nalepszy token z maski
def find_best_token(mask, p=0.9):
    # to ppb tak na serio to niepotrzebne
    ppbs = F.log_softmax(mask, dim=-1)
    return torch.argmax(ppbs, dim=-1).unsqueeze(0)

# tworzy trie do pliku
def create_and_save_tire(tokenizer, file_path):
    with open(file_path, 'r') as file:
        words = file.read().split()

        trie = Trie()
        encoded_words = tokenizer(words, padding=True, return_tensors="pt")

        # word to tensor z id tokenow slowa
        for word in tqdm(encoded_words['input_ids'], total=len(encoded_words['input_ids'])):
            try:
                index = (word == 2).nonzero(as_tuple=True)[0][0].item()
            except:
                index = len(word)
            word = word[1:index]
            trie.insert(word)

        with open('trie.pkl', 'wb') as file:
            pickle.dump(trie, file)


def generate_words_from_trie(trie, model, tokenizer, input_string, generation_num=2):
    # RETURNs words ids in a list sorted from most probable to least probable according to beam search approach
    
    # definitionsa
    node_for_each_generation = [trie.root for i in range(generation_num)]
    generated_words = [torch.tensor([]).to(device) for i in range(generation_num)]
    is_complete = [False for i in range(generation_num)]

    input_ids = tokenizer(input_string, return_tensors="pt")['input_ids'].to(device)

    # HERE FIRST GENERATION BEGINS - FROM THE SAME TRIE NODE

    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]

    # Create a tensor with allowed token indices
    allowed_tokens = torch.tensor(list(node_for_each_generation[0].children.keys())).to(device)

    # Mask tokens not in the current trie node
    mask = torch.full(next_token_logits.shape, float('-inf')).to(device)
    mask[0, allowed_tokens] = next_token_logits[0, allowed_tokens]

    # taking 'generation_num' best tokens
    for i in range(generation_num):
        next_token = find_best_token(mask)
        mask[0, next_token] = float('-inf')
        generated_words[i] = torch.cat([generated_words[i], next_token], dim=-1)
        node_for_each_generation[i] = Trie.find_next_node(node_for_each_generation[i], next_token)

        # is it the end of the word?
        if node_for_each_generation[i].is_word_end: is_complete[i] = True

    # HERE OTHER GENERATIONS BEGIN - FROM DIFFERENT TRIE NODES

    while not all(is_complete):
        for i in range(generation_num):
            if is_complete[i]: continue
            
            # preparing unfinished generation
            input_ids = torch.cat([input_ids, generated_words[i]], dim=-1).long()

            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # create a tensor with allowed token indices
            allowed_tokens = torch.tensor(list(node_for_each_generation[i].children.keys())).to(device)

            # mask tokens not in the current trie node
            mask = torch.full(next_token_logits.shape, float('-inf')).to(device)
            mask[0, allowed_tokens] = next_token_logits[0, allowed_tokens]
            
            # taking the best token
            next_token = find_best_token(mask)
            mask[0, next_token] = float('-inf')
            generated_words[i] = torch.cat([generated_words[i], next_token], dim=-1)
            node_for_each_generation[i] = Trie.find_next_node(node_for_each_generation[i], next_token) # moze trzeba next_token zrobic .item() ??

            # is it the end of the word?
            if node_for_each_generation[i] is None or node_for_each_generation[i].is_word_end:
                is_complete[i] = True
        
    return generated_words


print('Loading model...')
# model_name = 'eryk-mazus/polka-1.1b'
model_name = 'eryk-mazus/polka-1.1b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# create_and_save_tire(tokenizer, 'superbazy_clean.txt')
print('Loading trie...')
with open('trie.pkl', 'rb') as file:
    trie = pickle.load(file)

prompt = """Rozwiązuj poniższe zagadki: otrzymasz jednozdaniowy opis, a Twoim zadaniem jest podać pojedynczy wyraz, który najlepiej pasuje do opisu.

Przykład:
Zagadka: Woda, która spada z nieba.
Odpowiedź: deszcz

Zagadka: [zagadka]
Odpowiedź: """

print('Generating words...')
zagadka = 'rękopiśmienny tekst lub dokument, niepublikowany drukiem.'
# zagadka = 'postawa, przekonania lub działania mające na celu dyskryminację, prześladowanie lub nienawiść wobec żydów jako grupy etnicznej, religijnej lub kulturowej.'
prompt = prompt.replace('[zagadka]', zagadka.capitalize())
words = generate_words_from_trie(trie, model, tokenizer, prompt, generation_num=3)

print('Decoding words...')
for word in words:
    print(tokenizer.decode(word[0].int()))


def perform_test(test_file, trie=trie, model=model, tokenizer=tokenizer, prompt=prompt):
    with open(test_file, 'r') as file:
        lines = file.readlines()

    correct = 0
    # for line in lines:
    for line in tqdm(lines, total=len(lines)):
        answer, question = line.split(';;')
        answer, question = answer.strip(), question.strip().capitalize()
        new_prompt = prompt.replace('[zagadka]', question)
        words = generate_words_from_trie(trie, model, tokenizer, new_prompt, generation_num=3)
        decoded_words = [tokenizer.decode(word[0].int()) for word in words]
        if answer in decoded_words:
            correct += 1
        # print(f'{answer} -> {decoded_words}')
    print(f'Correct answers: {correct}/{len(lines)}')

# perform_test('/home/patryk/Downloads/zagadki_do_testow_clean.txt')