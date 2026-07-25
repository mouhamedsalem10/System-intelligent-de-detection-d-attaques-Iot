import os

files = [
    'data/train_test_networks.csv',
    'data/eval_test_full.csv'
]

for f in files:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as fp:
            line = fp.readline()
        semi  = line.count(';')
        comma = line.count(',')
        tab   = line.count('\t')
        print(f'Fichier : {f}')
        print(f'  virgules(,)        = {comma}')
        print(f'  points-virgules(;) = {semi}')
        print(f'  tabulations        = {tab}')
        if semi > comma:
            print(f'  --> Separateur : POINT-VIRGULE (;)')
        else:
            print(f'  --> Separateur : VIRGULE (,)')
        print()
    else:
        print(f'FICHIER INTROUVABLE : {f}')