import torch
from torch import nn
import math
from settings import max_seq_len


class Transformer(nn.Transformer):
    def fast_infer(self, src, src_mask=None, tgt_mask = None, memory_mask = None,
                   src_key_padding_mask = None, memory_key_padding_mask = None):
        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        if src.size(1) > 1:
            raise RuntimeError("only batch size 1 are allowed during inference")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        tgt = torch.zeros((1, 1, self.d_model), dtype=torch.float32)
        tgt_key_padding_mask = torch.tensor([[True]], dtype=torch.bool)
        for i in range(len(src)):
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
            tgt = torch.cat([tgt, output[-1:]], dim=0)
            tgt_key_padding_mask = torch.cat([tgt_key_padding_mask, torch.tensor([[False]], dtype=torch.bool)], dim=1)

        return tgt


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=max_seq_len):
        super().__init__()
        self.d_model = d_model
        pe = self.calc_pos()
        self.register_buffer('pe', pe)

    def calc_pos(self):
        seq_len = max_seq_len+1
        pe = torch.zeros(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)]
        return x

    def unforward(self, x):
        x = x - self.pe[:x.size(0)]
        x = x / math.sqrt(self.d_model)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.transformer = Transformer(d_model=80, dim_feedforward=512, nhead=8,
                                       num_encoder_layers=8, num_decoder_layers=8)
        self.lin = nn.Linear(80, 80)
        self.pe = PositionalEncoder(d_model=80)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input):
        src = self.pe(input['noisy'].cuda().transpose(1, 0))
        tgt = self.pe(input['clean'].cuda().transpose(1, 0))

        tgt_pad_mask = input['tgt_pad_mask'].cuda()
        src_pad_mask = input['src_pad_mask'].cuda()
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0))

        preds = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                 tgt_key_padding_mask=tgt_pad_mask)
        preds = self.lin(preds)

        loss = self.criterion(preds[:-1], input['clean'][1:])
        loss = loss * tgt_pad_mask[1:]
        loss = loss.sum(dim=0)/tgt_pad_mask[1:].sum(dim=0)
        loss = loss.mean()

        return loss

    def predict(self, input):
        src = self.pe(input['noisy'].cuda().transpose(1, 0))
        preds = self.transformer.fast_infer(src)
        preds = self.lin(preds)

        loss = None
        if input['clean']:
            loss = self.criterion(preds, input['clean'][1:]).sum()

        preds = self.pe.unforward(preds)

        return preds.transpose(1, 0), loss
