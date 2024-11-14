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

# word to tensor z id tokenow slowa
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


# def generate_word_from_trie(input_string)
#     input_ids = torch.tensor(input_string, return_tensor="pt")['input_ids'].to(device)
#     node = trie.root
#     generated_word = []

#     # for _ in range(max_length):
#     while True:
#         outputs = model(input_ids=input_ids)
#         next_token_logits = outputs.logits[:, -1, :]
        
#         # Create a tensor with allowed token indices
#         allowed_tokens = torch.tensor(list(node.children.keys())).to(device)
        
#         # Mask tokens not in the current trie node
#         mask = torch.full(next_token_logits.shape, float('-inf')).to(device)
#         mask[0, allowed_tokens] = next_token_logits[0, allowed_tokens]
        
#         next_token = torch.argmax(mask, dim=-1).unsqueeze(0)
#         input_ids = torch.cat([input_ids, next_token], dim=-1)
        
#         token_id = next_token.item()
#         generated_word.append(token_id)
        
#         node = trie.find_next_node(node, next_token)
#         if node is None or node.is_word_end:
#             break

#     return tokenizer.decode(generated_word, skip_special_tokens=True)

# # Example usage
# generated_word = generate_word_from_trie(model, tokenizer, trie)
# print(generated_word)