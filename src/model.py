import torch
from torch import nn
from einops import rearrange
import math
from torch import Tensor
import torchvision.models as models

from torch.nn import functional as F


class ICCModel(nn.Module):
    def __init__(self, device, pretrained, backbone, d_model, vocab_size, max_len,
                 num_heads, h_dim, a_dim, encoder_layers, decoder_layers, dropout,
                 learnable=False, fine_tune=True, tie_embeddings=True, prenorm=False):
        super(ICCModel, self).__init__()

        self.feature_dim = d_model
        visual = pretrained.visual if pretrained else None
        self.encoder = ImagesEncoder(device, visual, backbone, d_model, num_heads, h_dim, a_dim, dropout,
                                     encoder_layers, fine_tune)

        self.decoder = Decoder(device, d_model, vocab_size, max_len, num_heads,
                               decoder_layers, dropout,
                               learnable=learnable, tie_embeddings=tie_embeddings, prenorm=prenorm)

    def forward(self, img1, img2, input_ids, labels, attention_mask):
        vis_emb, vis_toks = self.encoder(img1, img2)
        cap_loss, text_emb, lm_logits, weights = self.decoder(input_ids, labels, attention_mask, vis_toks)
        return cap_loss, vis_emb, text_emb, vis_toks, lm_logits, weights


class ImagesEncoder(nn.Module):
    def __init__(self, device, pretrained, backbone, d_model, num_heads, h_dim, a_dim, dropout,
                 encoder_layers, fine_tune):
        super(ImagesEncoder, self).__init__()
        self.encoder = Encoder(pretrained, backbone, d_model, fine_tune)
        self.encoder_trans = AttentiveEncoder(device, encoder_layers,
                                              [self.encoder.feat_size, self.encoder.feat_size, d_model], num_heads,
                                              hidden_dim=h_dim, attention_dim=a_dim, dropout=dropout)

        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.Conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        self.LN = resblock(d_model, d_model)

        self.att_pool = nn.MultiheadAttention(d_model, num_heads)
        self.att_pool_norm = nn.LayerNorm(d_model)
        self.img_queries = nn.Parameter(torch.randn(1, d_model))

    def forward(self, img1, img2):
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        x1, x2 = self.encoder_trans(feat1, feat2)  # batch_size, channel, enc_image_size, enc_image_size

        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1)  # batch_size, 2channel, enc_image_size, enc_image_size
        x = self.LN(self.Conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)  # h*w, batch, dim

        img_queries = self.img_queries.unsqueeze(1).repeat(1, x.shape[1], 1)  # L,N,E
        img_emb = self.att_pool(img_queries, x, x, need_weights=False)[0]
        img_emb = self.att_pool_norm(img_emb)  # 1, batch, d_model

        cls = img_emb[0]
        return cls, x


class Encoder(nn.Module):
    def __init__(self, pretrained, backbone, d_model, fine_tune):
        super(Encoder, self).__init__()
        self.backbone = backbone

        if 'rn' in backbone.lower():
            modules = list(pretrained.children())[:-1]
            self.net = nn.Sequential(*modules)
            self.feat_dim = 2048
            self.feat_size = 7
        elif 'b-32' in backbone.lower():
            self.net = pretrained
            self.net.output_tokens = True
            self.feat_dim = 768
            self.feat_size = 7
        elif 'l-14' in backbone.lower():
            self.net = pretrained
            self.net.output_tokens = True
            self.feat_dim = 1024
            self.feat_size = 16
        elif backbone == 'resnet50':
            net = models.resnet50(pretrained=True)
            modules = list(net.children())[:-2]
            self.net = nn.Sequential(*modules)
            self.feat_dim = 2048
            self.feat_size = 8
        elif backbone == 'resnet101':
            net = models.resnet101(pretrained=True)
            modules = list(net.children())[:-2]
            self.net = nn.Sequential(*modules)
            self.feat_dim = 2048
            self.feat_size = 8

        self.proj = None
        if self.feat_dim != d_model:
            self.proj = nn.Conv2d(self.feat_dim, d_model, kernel_size=1)

        self.fine_tune(fine_tune)

    def forward(self, image):
        feat = self.net(image)  # batch, feat_dim, feat_size, feat_size
        if 'vit' in self.backbone.lower():
            feat = feat[1].reshape(-1, self.feat_size, self.feat_size, self.feat_dim).permute(0, 3, 1, 2)

        if self.proj:
            feat = self.proj(feat)

        return feat

    def fine_tune(self, fine_tune=True):
        for p in self.net.parameters():
            p.requires_grad = False

        if 'resnet' in self.backbone:
            to_finetune = list(self.net.children())[-5:]
        elif 'vit' in self.backbone.lower():
            to_finetune = list(self.net.children())[-2:]  # only transformer layers
        else:
            to_finetune = list(self.net.children())[-3:]  # only fine-tune convolutional blocks 2 through 4

        for c in to_finetune:
            for p in c.parameters():
                p.requires_grad = fine_tune


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads=8, dropout=0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias=False)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias=False)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2, x3):
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_k(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # (b,n,dim)


class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout=0., norm_first=False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + x1)
            x = self.norm2(self.feedforward(x) + x)

        return x


class AttentiveEncoder(nn.Module):
    def __init__(self, device, n_layers, feature_size, heads, hidden_dim=512, attention_dim=512, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size

        self.device = device
        self.h_embedding = nn.Embedding(h_feat, int(channels / 2))
        self.w_embedding = nn.Embedding(w_feat, int(channels / 2))
        self.selftrans = nn.ModuleList([])
        for i in range(n_layers):
            self.selftrans.append(nn.ModuleList([
                Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
                Transformer(channels * 2, channels * 2, heads, attention_dim, hidden_dim, dropout, norm_first=False),
            ]))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img1, img2):
        batch, c, h, w = img1.shape
        pos_h = torch.arange(h).to(self.device)
        pos_w = torch.arange(w).to(self.device)
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        img1 = img1 + pos_embedding
        img2 = img2 + pos_embedding
        img1 = img1.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
        img2 = img2.view(batch, c, -1).transpose(-1, 1)
        img_sa1, img_sa2 = img1, img2

        for (l, m) in self.selftrans:
            img_sa1 = l(img_sa1, img_sa1, img_sa1) + img_sa1
            img_sa2 = l(img_sa2, img_sa2, img_sa2) + img_sa2
            img = torch.cat([img_sa1, img_sa2], dim=-1)
            img = m(img, img, img)
            img_sa1 = img[:, :, :c] + img1
            img_sa2 = img[:, :, c:] + img2

        img1 = img_sa1.reshape(batch, h, w, c).transpose(-1, 1)
        img2 = img_sa2.reshape(batch, h, w, c).transpose(-1, 1)

        return img1, img2


class resblock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 2), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 2), int(outchannel / 2), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 2), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return F.relu(out)


class Decoder(nn.Module):
    def __init__(self, device, h_dim, vocab_size, max_len, n_head, n_layers, dropout=0.10,
                 learnable=False, tie_embeddings=True, prenorm=False):

        super(Decoder, self).__init__()

        self.embed_dim = h_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device

        self.tokens_embed = nn.Embedding(vocab_size, self.embed_dim)
        self.position_encoding = PositionalEncoding(self.embed_dim, dropout=dropout, max_len=max_len,
                                                    device=device, learnable=learnable)

        self.uni_decoder = nn.ModuleList(
            [DecoderLayer(h_dim, h_dim, n_head, dim_feedforward=h_dim * 4, dropout=self.dropout, prenorm=prenorm,
                          crossattention=False) for _ in range(n_layers)])

        self.cross_decoder = nn.ModuleList(
            [DecoderLayer(h_dim, h_dim, n_head, dim_feedforward=h_dim * 4, dropout=self.dropout, prenorm=prenorm,
                          crossattention=True) for _ in range(n_layers)])

        self.lm_head = nn.Linear(h_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.tokens_embed.weight = self.lm_head.weight
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()
        self.loss_fn = nn.CrossEntropyLoss()

    def init_weights(self):
        self.tokens_embed.weight.data.uniform_(-0.1, 0.1)
        self.lm_head.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_ids=None, labels=None, pad_mask=None, img_emb=None):
        att_weights = None
        mask = torch.tril(torch.ones(input_ids.shape[1], input_ids.shape[1]))
        mask = ~mask.bool()
        mask = mask.to(self.device)

        inputs_embeds = self.tokens_embed(input_ids)
        inputs_embeds = self.position_encoding(inputs_embeds)  # batch, seq, e_dim
        inputs_embeds = inputs_embeds.permute(1, 0, 2)  # seq, batch, e_dim

        # seq, batch, emb_dim
        out = inputs_embeds
        for block in self.uni_decoder:
            out, _ = block(out, None, tgt_mask=mask, tgt_key_padding_mask=pad_mask)

        if pad_mask is not None:  # not inference
            cls = []
            for i in range(pad_mask.shape[0]):
                end = pad_mask[i].shape[0] - pad_mask[i].count_nonzero()
                cls.append(out[end - 1, i, :])

            cls = torch.stack(cls)  # batch, emb_dim
        else:
            cls = None

        if img_emb is None:
            return None, cls, None, None

        for block in self.cross_decoder:
            out, att_weights = block(out, img_emb, tgt_mask=mask, tgt_key_padding_mask=pad_mask)

        lm_logits = self.lm_head(self.dropout(out))  # seq, batch, voc_dim
        lm_logits = lm_logits.permute(1, 0, 2)  # batch, seq, voc_dim

        if labels is not None:  # not inference
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            loss = self.loss_fn(shift_logits, shift_labels)
        else:
            loss = None

        return loss, cls, lm_logits, att_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, device, learnable=False):
        super(PositionalEncoding, self).__init__()
        self.learnable = learnable
        self.max_len = max_len
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

        if not learnable:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        else:
            self.pos_emb = nn.Embedding(max_len, int(d_model))

    def forward(self, x):
        if self.learnable:
            position_ids = torch.arange(x.size(1), dtype=torch.long).to(self.device)
            position_ids = position_ids.unsqueeze(0).view(-1, x.size(1))  # batch, seq
            x = x + self.pos_emb(position_ids)
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, img_dim, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5,
                 prenorm=False, crossattention=False):
        super(DecoderLayer, self).__init__()

        self.prenorm = prenorm
        self.crossattention = crossattention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        if crossattention:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=img_dim, vdim=img_dim)
            self.mha_dropout = nn.Dropout(dropout)
            self.mha_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ff_linear1 = nn.Linear(d_model, dim_feedforward)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_linear2 = nn.Linear(dim_feedforward, d_model)

        self.sa_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ff_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.sa_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        att_weight = None
        x = tgt

        if self.prenorm:
            x = x + self._sa_block(self.sa_norm(x), tgt_mask, tgt_key_padding_mask)
            if self.crossattention:
                enc_att, att_weight = self._mha_block(self.mha_norm(x), memory, memory_mask, memory_key_padding_mask)
                x = x + enc_att
            x = x + self._ff_block(self.ff_norm(x))
        else:
            x = self.sa_norm(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            if self.crossattention:
                enc_att, att_weight = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
                x = self.mha_norm(x + enc_att)

            x = self.ff_norm(x + self._ff_block(x))
        return x, att_weight

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,  # L,N,E
                           attn_mask=attn_mask,  # L, S
                           key_padding_mask=key_padding_mask,  # N, S
                           is_causal=True,
                           need_weights=False)[0]
        return self.sa_dropout(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x, att_weight = self.cross_attn(x, mem, mem,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask,
                                        is_causal=False,
                                        need_weights=True)
        return self.mha_dropout(x), att_weight

    def _ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout(self.activation(self.ff_linear1(x))))
        return self.ff_dropout(x)
