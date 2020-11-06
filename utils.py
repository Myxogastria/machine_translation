from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu


def load_dataset(dataname, encoding='utf-8'):
    # wget http://www.manythings.org/anki/jpn-eng.zip
    if dataname == 'jpn-eng':
        filename = '../jpn-eng/jpn.txt'
    elif dataname == 'deu-eng':
        filename = '../deu-eng/deu.txt'
    en_texts = []
    trans_texts = []
    with open(filename, encoding=encoding) as f:
        for line in f:
            en_text, trans_text = line.strip().split('\t')[:2]
            en_texts.append(en_text.lower())
            trans_texts.append(trans_text.lower())
    return en_texts, trans_texts


def evaluate_bleu(X, y, api):
    d = defaultdict(list)
    for source, target in zip(X, y):
        d[source].append(target)
    hypothesis = []
    references = []
    for source, targets in d.items():
        pred = api.predict(source)
        hypothesis.append(pred)
        references.append(targets)
    bleu_score = corpus_bleu(references, hypothesis)
    return bleu_score
