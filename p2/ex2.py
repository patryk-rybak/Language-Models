import torch
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
            
    @staticmethod
    def find_next_node(cls, starter_node, token_id):# item() ???
        return starter_node[token_id.item()]

model_name = 'eryk-mazus/polka-1.1b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

with open('superbazy_clean.txt', 'r') as file:
    words = file.read().split()

trie = Trie()

encoded_words = tokenizer(words, padding=True, return_tensors="pt")
for word in tqdm(encoded_words['input_ids'], total=len(encoded_words['input_ids'])):
    try:
        index = (word == 2).nonzero(as_tuple=True)[0][0].item()
    except:
        index = len(word)
    word = word[1:index]
    trie.insert(word)

import pickle
with open('trie.pkl', 'wb') as file:
    pickle.dump(trie, file)
# with open('trie_data.pkl', 'rb') as file:
#     trie = pickle.load(file)

