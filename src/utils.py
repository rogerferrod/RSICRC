import json
import os

import pandas as pd
import torch
import glob

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider


def get_eval_score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)

    score_dict = dict(zip(method, score))
    return score_dict


def get_vocabulary(in_path, out_file):
    if 'levir' in in_path.lower():
        return get_levir_vocabulary(in_path, out_file)
    elif 'dubai' in in_path.lower():
        return get_dubai_vocabulary(in_path, out_file)
    elif 'clevr' in in_path.lower():
        return get_clevr_vocabulary(in_path, out_file)


def get_levir_vocabulary(in_path, out_file):
    with open(in_path) as fin:
        data = json.load(fin)['images']

    sents = [y for x in data for y in x['sentences']]
    tokens = [y for x in sents for y in x['tokens']]
    occurencies = pd.Series(tokens).value_counts()
    selected = occurencies[occurencies > 5]
    vocab = {w: i + 4 for i, w in enumerate(selected.index)}
    vocab['PAD'] = 0
    vocab['START'] = 1
    vocab['UNK'] = 2
    vocab['END'] = 3

    with open(out_file, 'w') as fout:
        json.dump(vocab, fout)

    return vocab


def get_dubai_vocabulary(in_path, out_file):
    data = []
    for path in glob.glob(in_path + '/*.json'):
        with open(path) as fin:
            data.extend(json.load(fin)['images'])

    sents = [y for x in data for y in x['sentences']]
    tokens = [y for x in sents for y in x['tokens']]
    selected = pd.Series(tokens).value_counts()
    vocab = {w: i + 4 for i, w in enumerate(selected.index)}
    vocab['PAD'] = 0
    vocab['START'] = 1
    vocab['UNK'] = 2
    vocab['END'] = 3

    with open(out_file, 'w') as fout:
        json.dump(vocab, fout)

    return vocab


def get_clevr_vocabulary(in_path, out_file):
    sents = []
    with open(os.path.join(in_path, 'change_captions.json'), 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        sents += [y for x in data for y in data[x]]

    with open(os.path.join(in_path, 'no_change_captions.json'), 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        sents += [y for x in data for y in data[x]]

    tokens = [y for x in sents for y in x.split(' ')]
    occurencies = pd.Series(tokens).value_counts()
    vocab = {w: i + 4 for i, w in enumerate(occurencies.index)}
    vocab['PAD'] = 0
    vocab['START'] = 1
    vocab['UNK'] = 2
    vocab['END'] = 3

    with open(out_file, 'w') as fout:
        json.dump(vocab, fout)

    return vocab


def unormalize(tensor, mean=None, std=None):
    if mean is not None and std is not None:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clip(tensor, min=0, max=1)

    b, c, h, w = tensor.shape
    tensor = tensor.view(b, -1)
    tensor -= tensor.min(1, keepdim=True)[0]
    tensor /= tensor.max(1, keepdim=True)[0]
    return tensor.view(b, c, h, w)
