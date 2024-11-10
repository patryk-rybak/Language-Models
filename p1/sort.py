with open('task4_questions.txt', 'r') as dupa:
    lines = [l for l in dupa]
    lines.sort()
    for l in lines: print(l.strip())


