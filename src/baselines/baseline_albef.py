import argparse
import random
import os
from ruamel.yaml import YAML

import numpy as np
import torch
import faiss
from tqdm import tqdm
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

from DatasetsBaseline import CCDataset

AT_K = sorted([1, 3, 5, 10], reverse=True)
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def retrieve(args, model, tokenizer, dataset, transform, device):
    scores_p, scores_r, scores_rr = search(model, tokenizer, dataset, transform, args.sim, device)
    with open(os.path.join(args.output_path, 'retrieve.txt'), 'w') as out:
        for k in AT_K:
            out.write('P@{0} {1:.4f}\n'.format(k, scores_p[k]))
            out.write('R@{0} {1:.4f}\n'.format(k, scores_r[k]))
            out.write('MRR@{0} {1:.4f}\n'.format(k, scores_rr[k]))
            out.write('\n')

    scores_p, scores_r, scores_rr = search(model, tokenizer, dataset, transform, args.sim, device, sub=True)
    with open(os.path.join(args.output_path, 'retrieve_sub.txt'), 'w') as out:
        for k in AT_K:
            out.write('P@{0} {1:.4f}\n'.format(k, scores_p[k]))
            out.write('R@{0} {1:.4f}\n'.format(k, scores_r[k]))
            out.write('MRR@{0} {1:.4f}\n'.format(k, scores_rr[k]))
            out.write('\n')


@torch.no_grad()
def search(model, tokenizer, dataset, transform, threshold, device, sub=False):
    model.eval()

    visual = None
    textual = None
    flags = []
    embs = []

    for i in tqdm(range(len(dataset)), desc='Indexing'):
        image_before, image_after, text, flag, emb = dataset[i]
        if sub and flag == -1:
            continue

        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        txt = model.text_proj(text_feat[:, 0, :])  # 1, 256

        img_1 = transform(image_before).unsqueeze(0).to(device)
        img_2 = transform(image_after).unsqueeze(0).to(device)
        diff = torch.abs(img_2 - img_1)

        image_feat = model.visual_encoder(diff)
        img = model.vision_proj(image_feat[:, 0, :])  # 1, 256

        visual = torch.cat([visual, img.cpu()]) if visual is not None else img.cpu()
        textual = torch.cat([textual, txt.cpu()]) if textual is not None else txt.cpu()
        flags.append(flag)
        embs.append(torch.tensor(emb))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    index = faiss.IndexFlatIP(visual.shape[1])

    visual = F.normalize(visual, p=2, dim=1)
    textual = F.normalize(textual, p=2, dim=1)

    embs = torch.stack(embs).to(device)
    sims = torch.matmul(embs, torch.t(embs))

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


def run(args, config):
    print('Initializing...')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']

    if args.finetune:
        chkpt = torch.load(args.finetune, map_location='cpu')
        for k in chkpt:
            state_dict[k] = chkpt[k]

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    print('Loading...')
    dataset = CCDataset(args.annotation_json, args.image_dir, 'test',
                        'sentence-transformers/msmarco-distilbert-cos-v5', device)

    print('Final evaluation...')
    retrieve(args, model, tokenizer, dataset, transform, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--annotation_json', type=str, default='../../input/Levir_CC/LevirCCcaptions.json')
    parser.add_argument('--image_dir', type=str, default='../../input/Levir_CC/images/')
    parser.add_argument('--sim', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default='../../input/ALBEF.pth')
    parser.add_argument('--finetune', type=str)  # e.g., ../../input/flickr30k.pth
    parser.add_argument('--output_path', type=str, default='../../output/')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))
    run(args, config)


if __name__ == '__main__':
    main()
