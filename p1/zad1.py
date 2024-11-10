import json

with open('plik', 'r') as plik, open('keys.json', 'r') as keys:
    words_list = list()
    res = {}
    for line in plik:
        words = line.split()
        for w in words: words_list.append(w)

    s = json.load(keys)
    for key in s:
        value = [word for word in words_list if key in word]
        if len(value) == 0: continue
        res[key] = value

    print(json.dumps(res, indent=2))