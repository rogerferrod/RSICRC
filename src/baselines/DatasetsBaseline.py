import os
import json
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm


class CCDataset(Dataset):
    def __init__(self, json_file, root_dir, split, s_pretrained, device):
        super(CCDataset, self).__init__()
        self.split = split
        self.device = device
        assert self.split in {'test'}

        s_model = SentenceTransformer(s_pretrained)
        self.s_model = s_model.to(device)
        self.root_dir = root_dir

        with open(json_file) as f:
            data = json.load(f)['images']

        self.raw_dataset = [entry for entry in data if entry['split'] == split]

        self.embeddings = []
        for record in tqdm(self.raw_dataset, desc='Embeddings ' + self.split):
            self.embeddings.extend(self.compute_embeddings(record['sentences']))

    def compute_embeddings(self, batch):
        sents = [x['raw'].strip() for x in batch]
        embs = self.s_model.encode(sents)
        return embs

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        before_img_path = os.path.join(self.root_dir, self.raw_dataset[idx]['filepath'], 'A',
                                       self.raw_dataset[idx]['filename'])
        image_before = Image.open(before_img_path)
        after_img_path = os.path.join(self.root_dir, self.raw_dataset[idx]['filepath'], 'B',
                                      self.raw_dataset[idx]['filename'])
        image_after = Image.open(after_img_path)

        text = self.raw_dataset[idx]['sentences'][0]['raw']
        flag = -1 if self.raw_dataset[idx]['changeflag'] == 0 else self.raw_dataset[idx]['imgid']
        emb = self.embeddings[idx]
        return image_before, image_after, text, flag, emb
