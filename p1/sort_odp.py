with open('task4_answers.txt', 'r') as dupa:
    lines = [l for l in dupa]
    lines.sort()
    for l in lines: print(l.strip())


