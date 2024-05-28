import argparse
import random
import os
from datetime import datetime

import numpy as np
import torch
import json
from torch.optim import AdamW
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from Datasets import CCDataset, Batcher
from model import ICCModel
from utils import get_vocabulary
from Loss import InfoNCELoss
from eval import captioning, retrieve, plot
from huggingface_hub import hf_hub_download
import open_clip


def train(args, model, train_loader, valid_loader, device, infonce, optim, scheduler, writer):
    step = 0
    best_score = float("inf")
    best_model = None

    for epoch in range(args.epochs):
        model.train()

        for batch in tqdm(train_loader, desc='Epoch ' + str(epoch)):
            imgs1 = batch['images_before'].to(device)
            imgs2 = batch['images_after'].to(device)
            toks = batch['input_ids'].to(device)
            labs = batch['labels'].to(device)
            flags = batch['flags'].to(device)
            attention_mask = batch['pad_mask'].to(device)
            embs = batch['embs'].to(device)

            cap_loss, vis_emb, text_emb, _, _, _ = model(imgs1, imgs2, toks, labs, attention_mask)

            con_loss, num_pos = infonce(vis_emb, text_emb, flags, embs)
            loss = cap_loss + args.lamb * con_loss
            loss.backward()

            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad = torch.norm(torch.stack(
                [torch.norm(p.grad.detach()).to(device) for p in model.parameters() if p.grad is not None]))

            optim.step()
            scheduler.step()
            optim.zero_grad()

            writer.add_scalar('train_loss', loss.item(), step)
            writer.add_scalar('grad', grad, step)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

            step += 1

        torch.save(model.state_dict(), args.output_path + 'model_{}.pt'.format(step))

        model.eval()
        with torch.no_grad():
            eval_losses = torch.empty(0)
            for batch in tqdm(valid_loader, desc='Validation ' + str(epoch)):
                imgs1 = batch['images_before'].to(device)
                imgs2 = batch['images_after'].to(device)
                toks = batch['input_ids'].to(device)
                labs = batch['labels'].to(device)
                flags = batch['flags'].to(device)
                attention_mask = batch['pad_mask'].to(device)
                embs = batch['embs'].to(device)

                cap_loss, vis_emb, text_emb, _, _, _ = model(imgs1, imgs2, toks, labs, attention_mask)

                con_loss, _ = infonce(vis_emb, text_emb, flags, embs)
                loss = cap_loss + args.lamb * con_loss
                eval_losses = torch.cat([eval_losses, loss.cpu().unsqueeze(0)])

            eval_score = torch.mean(eval_losses)
            writer.add_scalar('eval_score', eval_score, step)

        is_best = eval_score < best_score
        best_score = min(eval_score, best_score)
        if is_best:
            best_model = step

    if best_model is not None:
        state_dict = torch.load(os.path.join(args.output_path + 'model_{}.pt'.format(best_model)), map_location=device)
        torch.save(state_dict, args.output_path + 'model_best.pt')


def run(args, config):
    print('Initializing...')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    dt_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    writer_path = args.output_path + dt_str
    writer = SummaryWriter(writer_path)

    if os.path.exists(args.vocab):
        with open(args.vocab, 'r') as infile:
            vocab = json.load(infile)
    else:
        vocab = get_vocabulary(args.annotation_json, args.vocab)

    clip = None
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if 'resnet' not in config['backbone']:
        checkpoint_path = hf_hub_download("chendelong/RemoteCLIP",
                                          f"RemoteCLIP-{config['backbone']}.pt",
                                          cache_dir=args.pretrained)

        clip, _, preprocess = open_clip.create_model_and_transforms(config['backbone'])
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        clip.load_state_dict(ckpt)

    model = ICCModel(device, clip, config['backbone'], config['d_model'],
                     len(vocab), config['max_len'], config['num_heads'], config['h_dim'], config['a_dim'],
                     config['encoder_layers'], config['decoder_layers'], config['dropout'],
                     learnable=config['learnable'], fine_tune=config['fine_tune'],
                     tie_embeddings=config['tie_embeddings'], prenorm=config['prenorm'])
    model = model.to(device)
    del clip

    print('Loading...')
    training_set = CCDataset(args.annotation_json, args.image_dir, vocab, preprocess, 'train', config['max_len'],
                             config['s-transformers'], device)
    valid_set = CCDataset(args.annotation_json, args.image_dir, vocab, preprocess, 'val', config['max_len'],
                          config['s-transformers'], device)
    test_set = CCDataset(args.annotation_json, args.image_dir, vocab, preprocess, 'test', config['max_len'],
                         config['s-transformers'], device)

    train_loader = Batcher(training_set, args.batch_size, config['max_len'], device, args.hd, model=model, shuffle=True)
    valid_loader = Batcher(valid_set, args.batch_size, config['max_len'], device)
    test_loader = Batcher(test_set, 1, config['max_len'], device)

    print('Training...')
    infonce = InfoNCELoss(device, k=args.k, temperature=args.temperature, threshold=config['s-threshold'],
                          fna=config['fna'])
    optim = AdamW([x for x in model.parameters() if x.requires_grad], lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_constant_schedule_with_warmup(optim,
                                                  num_warmup_steps=args.warmup_steps * len(train_loader) * args.epochs)
    train(args, model, train_loader, valid_loader, device, infonce, optim, scheduler, writer)

    print('Final evaluation...')
    model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_best.pt'), map_location=device))
    results = captioning(args, config, model, test_loader, vocab, device)
    retrieve(args, config, model, test_loader, device)
    plot(args, model.encoder.encoder.feat_size, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json', type=str, default='../input/Levir_CC/LevirCCcaptions.json')
    parser.add_argument('--image_dir', type=str, default='../input/Levir_CC/images/')
    parser.add_argument('--vocab', type=str, default='../input/levir_vocab.json')
    parser.add_argument('--pretrained', type=str, default='../../input/checkpoints')
    parser.add_argument('--config', type=str, default='../config.json')
    parser.add_argument('--output_path', type=str, default='../output/')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--k', type=int, default=-1)
    parser.add_argument('--hd', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=float, default=0.025)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    run(args, config)


if __name__ == '__main__':
    main()
