import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import faiss
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torchvision.transforms as transforms
from random import choice


class CCDataset(Dataset):
    def __init__(self, json_file, root_dir, vocab, transform, split, max_length, s_pretrained, device):
        super(CCDataset, self).__init__()
        self.vocab = vocab
        self.split = split
        self.max_length = max_length
        self.device = device
        self.transform = transform
        assert self.split in {'train', 'val', 'test'}

        s_model = SentenceTransformer(s_pretrained)
        self.s_model = s_model.to(device)

        self.root_dir = root_dir
        self.convert = transforms.ToTensor()

        with open(json_file) as f:
            data = json.load(f)['images']

        self.raw_dataset = [entry for entry in data if entry['split'] == split]
        self.sentences = []
        self.embeddings = []

        self.images = []
        self.captions = []
        for record in tqdm(self.raw_dataset, desc='Tokenize ' + self.split):
            self.sentences.extend(self.tokenize(record['sentences']))

        for record in tqdm(self.raw_dataset, desc='Embeddings ' + self.split):
            self.embeddings.extend(self.compute_embeddings(record['sentences']))

        self.preprocess()
        del self.raw_dataset
        del self.sentences
        del self.embeddings
        del self.s_model

    def tokenize(self, batch):
        for elem in batch:
            tokens = [self.vocab[x] if x in self.vocab.keys() else self.vocab['UNK'] for x in elem['tokens']]
            if len(tokens) > self.max_length - 2:
                continue

            tokens = [self.vocab['START']] + tokens + [self.vocab['END']]

            mask = [False] * len(tokens)

            diff = self.max_length - len(tokens)
            tokens += [self.vocab['PAD']] * diff
            mask += [True] * diff  # True = pad

            elem['input_ids'] = tokens
            elem['mask'] = mask

        if len(batch) < 5:
            diff = 5 - len(batch)
            batch += [choice(batch) for _ in range(diff)]

        assert len(batch) == 5
        return batch

    def compute_embeddings(self, batch):
        sents = [x['raw'].strip() for x in batch]
        embs = self.s_model.encode(sents)
        return embs

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_idx = idx // 5 if self.split == 'train' else idx
        elem = self.captions[idx]
        for k, v in self.images[img_idx].items():
            elem[k] = v
        return elem

    def preprocess(self):
        idx = 0
        prev_idx = -1
        pbar = tqdm(total=len(self.sentences), desc='Preprocessing ' + self.split)
        while idx < len(self.sentences):
            img_idx = idx // 5
            assert (self.sentences[idx]['imgid'] == self.raw_dataset[img_idx]['imgid'])

            input_ids = torch.tensor(self.sentences[idx]['input_ids'], dtype=torch.long)
            mask = torch.tensor(self.sentences[idx]['mask'], dtype=torch.bool)
            raws = [x['raw'] for x in self.raw_dataset[img_idx]['sentences']]
            flag = -1 if self.raw_dataset[img_idx]['changeflag'] == 0 else self.raw_dataset[img_idx]['imgid']
            flag = torch.tensor(flag, dtype=torch.long)
            embs = torch.tensor(self.embeddings[idx]) if len(self.embeddings) > 0 else None

            self.captions.append({'input_ids': input_ids, 'pad_masks': mask, 'raws': raws, 'flags': flag, 'embs': embs})

            if img_idx != prev_idx:
                before_img_path = os.path.join(self.root_dir, self.raw_dataset[img_idx]['filepath'], 'A',
                                               self.raw_dataset[img_idx]['filename'])
                image_before = Image.open(before_img_path)
                after_img_path = os.path.join(self.root_dir, self.raw_dataset[img_idx]['filepath'], 'B',
                                              self.raw_dataset[img_idx]['filename'])
                image_after = Image.open(after_img_path)

                image_before = self.transform(image_before).unsqueeze(0)
                image_after = self.transform(image_after).unsqueeze(0)

                self.images.append({'image_before': image_before, 'image_after': image_after, 'flags': flag})
                prev_idx = img_idx

            inc = 1 if self.split == 'train' else 5
            idx += inc
            pbar.update(inc)

        pbar.close()


class Batcher:
    def __init__(self, dataset, batch_size, max_len, device, hd=0, model=None, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.hd = hd
        self.max_len = max_len
        self.device = device
        self.model = model
        self.index = None
        self.visual = None
        self.textual = None

        self.ptr = 0
        self.indices = np.arange(len(self.dataset))
        self.shuffle = shuffle

        if shuffle:
            np.random.shuffle(self.indices)

        if model and hd > 0 and self.dataset.split == 'train':
            self.create_index()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __next__(self):
        if self.ptr >= len(self.dataset):
            self.ptr = 0
            self.index = None
            self.visual = None
            self.textual = None

            if self.shuffle:
                np.random.shuffle(self.indices)
            if self.model and self.hd > 0 and self.dataset.split == 'train':
                self.create_index()

            raise StopIteration

        batched = 0
        samples = []
        hard_negatives = []
        while self.ptr < len(self.dataset) and batched < self.batch_size:
            sample = self.dataset[self.indices[self.ptr]]
            samples.append(sample)

            if self.hd > 0 and self.dataset.split == 'train':
                hard_neg = self.mine_negatives(self.indices[self.ptr], self.hd)
                hard_negatives.extend(hard_neg)

            self.ptr += 1
            batched += 1

        return self.create_batch(samples + hard_negatives)

    def get_elem(self, idx):
        return self.dataset[idx]

    @torch.no_grad()
    def create_index(self):
        is_training = self.model.training
        self.model.eval()
        self.index = faiss.IndexFlatIP(self.model.feature_dim)
        prev_img = None
        for idx in tqdm(range(len(self.dataset)), desc='Creating index'):
            sample = self.dataset[idx]
            imgs1, imgs2, = sample['image_before'], sample['image_after']
            input_ids, mask = sample['input_ids'], sample['pad_masks']

            if idx // 5 != prev_img:
                imgs1 = imgs1.to(self.device)
                imgs2 = imgs2.to(self.device)
                vis_emb, _, = self.model.encoder(imgs1, imgs2)
                self.visual = torch.cat([self.visual, vis_emb.cpu()]) if self.visual is not None else vis_emb.cpu()
                prev_img = prev_img + 1 if prev_img is not None else 0

            input_ids = input_ids.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            _, text_emb, _, _ = self.model.decoder(input_ids, None, mask, None)
            self.textual = torch.cat([self.textual, text_emb.cpu()]) if self.textual is not None else text_emb.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.visual = F.normalize(self.visual, p=2, dim=1)
        self.textual = F.normalize(self.textual, p=2, dim=1)
        self.index.add(self.visual)
        if is_training:
            self.model.train()

    def mine_negatives(self, idx, n):
        negatives = []
        m = 4
        label = self.dataset[idx]['flags'].item()

        while len(negatives) < n and (n * m) < self.index.ntotal:
            k = n * m
            indeces = self.index.search(self.textual[idx].unsqueeze(0), k)[1][0]
            indeces = [x * 5 for x in indeces]
            negatives = [self.dataset[x] for x in indeces if self.dataset[x]['flags'].item() != label][:n]
            m *= 2

        return negatives

    def create_batch(self, samples):
        images_before = images_after = input_ids = pad_mask = labels = flags = embs = None
        raws = []

        for sample in samples:
            img1 = sample['image_before']
            img2 = sample['image_after']

            tokens = sample['input_ids']
            mask = sample['pad_masks']
            flag = sample['flags']
            emb = sample['embs']

            tokens = tokens.unsqueeze(0)
            mask = mask.unsqueeze(0)
            flag = flag.unsqueeze(0)
            lab = tokens.clone() * ~mask
            lab += torch.tensor([[-100]], dtype=torch.long).repeat(1, self.max_len) * mask
            if emb is not None:
                emb = emb.unsqueeze(0)

            images_before = torch.cat([images_before, img1]) if images_before is not None else img1
            images_after = torch.cat([images_after, img2]) if images_after is not None else img2
            input_ids = torch.cat([input_ids, tokens]) if input_ids is not None else tokens
            labels = torch.cat([labels, lab]) if labels is not None else lab
            pad_mask = torch.cat([pad_mask, mask]) if pad_mask is not None else mask
            flags = torch.cat([flags, flag]) if flags is not None else flag
            if emb is not None:
                embs = torch.cat([embs, emb]) if embs is not None else emb

            raws.append(sample['raws'])

        return {'images_before': images_before, 'images_after': images_after, 'input_ids': input_ids,
                'pad_mask': pad_mask, 'labels': labels, 'flags': flags, 'raws': raws, 'embs': embs}
