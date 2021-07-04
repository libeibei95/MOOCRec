"""
Script defining the model's architecture

@author: Abinash Sinha
"""

import numpy as np
import torch
import torch.nn as nn
from modules import DisentangledEncoder, SASEncoder, LayerNorm


class EduRecModel(nn.Module):
    def __init__(self, args):
        super(EduRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.sas_encoder = SASEncoder(args)
        self.disentangled_encoder = DisentangledEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def seq2seqloss(self, inp_subseq_encodings, label_subseq_encodings):
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)
        normalized_dot_product = torch.sum(product, dim=-1) / np.sqrt(self.args.hidden_size)
        numerator = torch.exp(normalized_dot_product)
        seq2seq_loss_k = -torch.log2(numerator / torch.sum(numerator))
        thresh = np.floor(self.args.lambda_ * self.args.pre_batch_size * self.args.num_intents)
        conf_indicator = seq2seq_loss_k <= thresh
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, conf_indicator)
        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)

        return seq2seq_loss

    def seq2itemloss(self, inp_subseq_encodings, next_item_emb):
        next_item_emb = torch.transpose(next_item_emb, 1, 2)
        dot_product = torch.matmul(inp_subseq_encodings, next_item_emb)  # [B, K, 1]
        exp_normalized_dot_product = torch.exp(dot_product / np.sqrt(self.args.hidden_size))
        numerator = torch.max(exp_normalized_dot_product, dim=1)[0]  # [B, 1]
        seq2item_loss_k = -torch.log2(numerator / torch.sum(exp_normalized_dot_product))  # [B, 1]
        seq2item_loss = torch.sum(seq2item_loss_k)

        return seq2item_loss

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def pretrain(self, inp_pos_items, label_pos_items, next_pos_item):

        next_item_emb = self.item_embeddings(next_pos_item)  # [B, 1, D]

        # Encode masked sequence
        inp_sequence_emb = self.add_position_embedding(inp_pos_items)
        inp_sequence_mask = (inp_pos_items == 0).float() * -1e8
        inp_sequence_mask = torch.unsqueeze(torch.unsqueeze(inp_sequence_mask, 1), 1)

        label_sequence_emb = self.add_position_embedding(label_pos_items)
        label_sequence_mask = (label_pos_items == 0).float() * -1e8
        label_sequence_mask = torch.unsqueeze(torch.unsqueeze(label_sequence_mask, 1), 1)

        inp_seq_encodings = self.disentangled_encoder(True,
                                                      inp_sequence_emb,
                                                      inp_sequence_mask,
                                                      output_all_encoded_layers=True)

        label_seq_encodings = self.disentangled_encoder(False,
                                                        label_sequence_emb,
                                                        label_sequence_mask,
                                                        output_all_encoded_layers=True)

        # seq2item loss
        seq2item_loss = self.seq2itemloss(inp_seq_encodings, next_item_emb)

        # seq2seq loss
        seq2seq_loss = self.seq2seqloss(inp_seq_encodings, label_seq_encodings)

        return seq2item_loss, seq2seq_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.sas_encoder(sequence_emb,
                                               extended_attention_mask,
                                               output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
