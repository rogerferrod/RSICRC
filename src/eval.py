import argparse
import random
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import json
import faiss
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T

import open_clip

from Datasets import CCDataset, Batcher
from model import ICCModel
from utils import get_vocabulary, unormalize, get_eval_score

AT_K = sorted([1, 3, 5, 10], reverse=True)


def captioning(args, config, model, data_loader, vocab, device):
    scores, results = inference(config, model, data_loader, vocab, device, return_results=True)

    with open(os.path.join(args.output_path, 'caption.txt'), 'w') as out:
        for t in scores.items():
            out.write(str(t) + '\n')

    scores, _ = inference(config, model, data_loader, vocab, device, sub=True, return_results=False)

    with open(os.path.join(args.output_path, 'caption_sub.txt'), 'w') as out:
        for t in scores.items():
            out.write(str(t) + '\n')

    return results


def retrieve(args, config, model, src_loader, device):
    scores_p, scores_r, scores_rr = search(config, model, src_loader, device)
    with open(os.path.join(args.output_path, 'retrieve.txt'), 'w') as out:
        for k in AT_K:
            out.write('P@{0} {1:.4f}\n'.format(k, scores_p[k]))
            out.write('R@{0} {1:.4f}\n'.format(k, scores_r[k]))
            out.write('MRR@{0} {1:.4f}\n'.format(k, scores_rr[k]))
            out.write('\n')

    scores_p, scores_r, scores_rr = search(config, model, src_loader, device, sub=True)
    with open(os.path.join(args.output_path, 'retrieve_sub.txt'), 'w') as out:
        for k in AT_K:
            out.write('P@{0} {1:.4f}\n'.format(k, scores_p[k]))
            out.write('R@{0} {1:.4f}\n'.format(k, scores_r[k]))
            out.write('MRR@{0} {1:.4f}\n'.format(k, scores_rr[k]))
            out.write('\n')


@torch.no_grad()
def search(config, model, src_loader, device, sub=False):
    model.eval()

    visual = None
    textual = None
    flags = []
    embs = None
    index = faiss.IndexFlatIP(config['d_model'])

    batcher = src_loader

    for batch in tqdm(batcher, desc='Indexing'):
        imgs1, imgs2, = batch['images_before'], batch['images_after']
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        flag = batch['flags']
        emb = batch['embs']
        if sub and flag[0] == -1:
            continue

        flags.append(flag)
        embs = torch.cat([embs, emb]) if embs is not None else emb

        vis_emb, _ = model.encoder(imgs1, imgs2)
        visual = torch.cat([visual, vis_emb.cpu()]) if visual is not None else vis_emb.cpu()

        input_ids, mask = batch['input_ids'], batch['pad_mask']
        input_ids = input_ids.to(device)
        mask = mask.to(device)

        _, text_emb, _, _ = model.decoder(input_ids, None, mask, None)
        textual = torch.cat([textual, text_emb.cpu()]) if textual is not None else text_emb.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embs = embs.to(device)
    sims = torch.matmul(embs, torch.t(embs))

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

        relevants = set(
            [x for x in range(len(textual)) if flags[x] == query_lab or sims[i][x] >= config['s-threshold']])

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


@torch.no_grad()
def inference(config, model, data_loader, vocab, device, sub=False, return_results=False):
    results = []
    references = []
    hypotheses = []
    inverse_vocab = {v: k for k, v in vocab.items()}

    model.eval()
    for batch in tqdm(data_loader, desc='Inference'):
        img1 = batch['images_before'][0].unsqueeze(0).to(device)
        img2 = batch['images_after'][0].unsqueeze(0).to(device)
        raws = batch['raws']
        flags = batch['flags']
        if sub and flags[0] == -1:
            continue

        references.append(raws[0])

        input_ids = torch.tensor([[vocab['START']]], dtype=torch.long, device=device)
        _, vis_toks = model.encoder(img1, img2)

        for _ in range(config['max_len']):
            _, _, lm_logits, weights = model.decoder(input_ids, None, None, vis_toks)

            next_item = lm_logits[0][-1].topk(1)[1]
            input_ids = torch.cat([input_ids, next_item.reshape(1, -1)], dim=1)
            if next_item.item() == vocab['END']:
                break

        words = [inverse_vocab[x] for x in input_ids[0].cpu().tolist()]
        sentence = ' '.join(words[1:-1]).strip()
        hypotheses.append([sentence])

        if return_results:
            results.append(
                (img1.cpu(), img2.cpu(), weights.detach().cpu(), vis_toks.detach().cpu(), sentence))

    score_dict = get_eval_score(references, hypotheses)
    return score_dict, results


def plot(args, feat_size, results):
    fig_idx = 0
    for img1, img2, weights, diff, sentence in tqdm(results, desc='Plot'):
        img1 = unormalize(img1)
        img1 = img1[0].permute(1, 2, 0)  # h,w,c
        img2 = unormalize(img2)
        img2 = img2[0].permute(1, 2, 0)  # h,w,c

        transform = T.Resize(size=(img1.size(0), img1.size(1)))
        weights = weights[0].reshape(-1, feat_size, feat_size)
        weights = transform(weights).permute(1, 2, 0)  # h,w,d
        weights = torch.sum(weights, 2) / weights.shape[2]
        after = img2  # h,w,c

        feature_map = diff[:, 0, :].reshape(-1, feat_size, feat_size)  # e,h,w
        feature_map = transform(feature_map).permute(1, 2, 0)  # h,w,c
        feature_map = torch.sum(feature_map, 2) / feature_map.shape[2]  # h, w

        fig, ax = plt.subplots(2, 2, figsize=(6, 8))
        fig.tight_layout()
        ax[0, 0].imshow(img1)
        ax[0, 0].set_title("Before")
        ax[0, 0].axis('off')
        ax[0, 1].imshow(img2)
        ax[0, 1].set_title("After")
        ax[0, 1].axis('off')

        ax[1, 0].set_title("Img diff")
        ax[1, 0].imshow(feature_map)
        ax[1, 0].axis('off')

        ax[1, 1].set_title("Att weights")
        ax[1, 1].imshow(after, interpolation='nearest')
        ax[1, 1].imshow(weights, interpolation='bilinear', alpha=0.5)
        ax[1, 1].axis('off')

        fig.text(.1, .05, sentence, wrap=True)

        with open(os.path.join(args.output_path, str(fig_idx) + '.png'), 'wb') as f:
            plt.savefig(f)
            plt.close()
            fig_idx += 1


def run(args, config):
    print('Initializing...')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if os.path.exists(args.vocab):
        with open(args.vocab, 'r') as infile:
            vocab = json.load(infile)
    else:
        vocab = get_vocabulary(args.annotation_json, args.vocab)

    clip, _, preprocess = open_clip.create_model_and_transforms(config['backbone'])

    model = ICCModel(device, clip, config['backbone'], config['d_model'],
                     len(vocab), config['max_len'], config['num_heads'], config['h_dim'], config['a_dim'],
                     config['encoder_layers'], config['decoder_layers'], config['dropout'],
                     learnable=config['learnable'], fine_tune=config['fine_tune'],
                     tie_embeddings=config['tie_embeddings'], prenorm=config['prenorm'])

    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    del clip

    print('Loading...')
    test_set = CCDataset(args.annotation_json, args.image_dir, vocab, preprocess, 'test', config['max_len'],
                         config['s-transformers'], device)
    test_loader = Batcher(test_set, 1, config['max_len'], device)

    print('Final evaluation...')
    results = captioning(args, config, model, test_loader, vocab, device)
    retrieve(args, config, model, test_loader, device)
    plot(args, model.encoder.encoder.feat_size, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../input/model_best.pt')
    parser.add_argument('--annotation_json', type=str, default='../input/Levir_CC/LevirCCcaptions.json')
    parser.add_argument('--image_dir', type=str, default='../input/Levir_CC/images/')
    parser.add_argument('--vocab', type=str, default='../input/levir_vocab.json')

    parser.add_argument('--config', type=str, default='../config.json')
    parser.add_argument('--output_path', type=str, default='../output/')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    run(args, config)


if __name__ == '__main__':
    main()
