import argparse
import random
import os

import numpy as np
import faiss
from tqdm import tqdm
import torch
import torch.nn.functional as F

import open_clip
from huggingface_hub import hf_hub_download
from DatasetsBaseline import CCDataset

AT_K = sorted([1, 3, 5, 10], reverse=True)


def retrieve(args, model, tokenizer, preprocess, dataset, device):
    scores_p, scores_r, scores_rr = search(model, tokenizer, preprocess, dataset, args.sim, device)
    with open(os.path.join(args.output_path, 'retrieve.txt'), 'w') as out:
        for k in AT_K:
            out.write('P@{0} {1:.4f}\n'.format(k, scores_p[k]))
            out.write('R@{0} {1:.4f}\n'.format(k, scores_r[k]))
            out.write('MRR@{0} {1:.4f}\n'.format(k, scores_rr[k]))
            out.write('\n')

    scores_p, scores_r, scores_rr = search(model, tokenizer, preprocess, dataset, args.sim, device, sub=True)
    with open(os.path.join(args.output_path, 'retrieve_sub.txt'), 'w') as out:
        for k in AT_K:
            out.write('P@{0} {1:.4f}\n'.format(k, scores_p[k]))
            out.write('R@{0} {1:.4f}\n'.format(k, scores_r[k]))
            out.write('MRR@{0} {1:.4f}\n'.format(k, scores_rr[k]))
            out.write('\n')


@torch.no_grad()
def search(model, tokenizer, preprocess, dataset, threshold, device, sub=False):
    model.eval()

    visual = None
    textual = None
    flags = []
    embs = []

    for i in tqdm(range(len(dataset)), desc='Indexing'):
        image_before, image_after, text, flag, emb = dataset[i]
        if sub and flag == -1:
            continue

        toks = tokenizer(text).to(device)
        img_1 = preprocess(image_before).unsqueeze(0).to(device)
        img_2 = preprocess(image_after).unsqueeze(0).to(device)

        diff = torch.abs(img_2 - img_1)

        img = model.encode_image(diff)
        txt = model.encode_text(toks)

        visual = torch.cat([visual, img.cpu()]) if visual is not None else img.cpu()
        textual = torch.cat([textual, txt.cpu()]) if textual is not None else txt.cpu()
        flags.append(flag)
        embs.append(torch.tensor(emb))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embs = torch.stack(embs).to(device)
    sims = torch.matmul(embs, torch.t(embs))

    index = faiss.IndexFlatIP(visual.shape[1])

    visual = F.normalize(visual, p=2, dim=1)
    textual = F.normalize(textual, p=2, dim=1)

    index.add(visual)

    scores_p = {k: [] for k in AT_K}
    scores_r = {k: [] for k in AT_K}
    scores_rr = {k: [] for k in AT_K}

    for i in tqdm(range(textual.shape[0]), desc='Ranking'):
        indices = None
        query = textual[i]
        query_lab = flags[i]

        relevants = set([x for x in range(len(textual)) if flags[x] == query_lab or sims[i][x] >= threshold])

        for k in AT_K:
            p = 0
            r = 0
            rr = 0

            if indices is None:
                indices = index.search(query.unsqueeze(0), k)[1][0]
            else:
                indices = indices[:k]

            for rank, idx in enumerate(indices):
                if idx in relevants:
                    if p == 0:
                        rr = 1 / (rank + 1)
                    p += 1
                    r += 1

            scores_p[k].append(p / len(indices))
            scores_r[k].append(r / len(relevants))
            scores_rr[k].append(rr)

    for k in AT_K:
        scores_p[k] = sum(scores_p[k]) / len(scores_p[k])
        scores_r[k] = sum(scores_r[k]) / len(scores_r[k])
        scores_rr[k] = sum(scores_rr[k]) / len(scores_rr[k])

    return scores_p, scores_r, scores_rr


def run(args):
    print('Initializing...')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP",
                                      f"RemoteCLIP-{args.model}.pt",
                                      cache_dir=args.checkpoints)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)

    print('Loading...')
    dataset = CCDataset(args.annotation_json, args.image_dir, 'test',
                        'sentence-transformers/msmarco-distilbert-cos-v5', device)

    print('Final evaluation...')
    retrieve(args, model, tokenizer, preprocess, dataset, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['RN50', 'ViT-B-32', 'ViT-L-14'], default='RN50')
    parser.add_argument('--annotation_json', type=str, default='../../input/Levir_CC/LevirCCcaptions.json')
    parser.add_argument('--image_dir', type=str, default='../../input/Levir_CC/images/')
    parser.add_argument('--sim', type=float, default=1.0)
    parser.add_argument('--checkpoints', type=str, default='../../input/checkpoints')
    parser.add_argument('--output_path', type=str, default='../../output/')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
