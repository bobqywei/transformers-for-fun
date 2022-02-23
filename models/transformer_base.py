from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MultiHeadAttention, LayerNormalization, FFN, PositionalEncoding


class EncoderBlock(nn.Module):

    def __init__(self, config: Dict):
        super(EncoderBlock, self).__init__()
        self.config = config
        dim_model = self.config["dim_model"]

        self.attention_layer = MultiHeadAttention(
            dim_model,
            self.config["model_num_heads"],
            self.config["model_dim_kq"],
            self.config["model_dim_v"])

        self.ffn = FFN(dim_model, self.config["model_ffn_dim_inner"])
        self.layer_norm_1 = LayerNormalization([dim_model])
        self.layer_norm_2 = LayerNormalization([dim_model])

        self.dropout = nn.Dropout(self.config["dropout_p"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention_layer(x, x, x)
        attn_out = self.dropout(attn_out)
        ffn_in = self.layer_norm_1(x + attn_out)

        ffn_out = self.ffn(ffn_in)
        ffn_out = self.dropout(ffn_out)
        out = self.layer_norm_2(ffn_in + ffn_out)

        return out


class DecoderBlock(nn.Module):

    def __init__(self, config: Dict):
        super(DecoderBlock, self).__init__()
        self.config = config
        dim_model = self.config["dim_model"]

        self.masked_attention = MultiHeadAttention(
            dim_model,
            self.config["model_num_heads"],
            self.config["model_dim_kq"],
            self.config["model_dim_v"])

        self.attention = MultiHeadAttention(
            dim_model,
            self.config["model_num_heads"],
            self.config["model_dim_kq"],
            self.config["model_dim_v"])

        self.ffn = FFN(dim_model, self.config["model_ffn_dim_inner"])
        self.layer_norm_1 = LayerNormalization([dim_model])
        self.layer_norm_2 = LayerNormalization([dim_model])
        self.layer_norm_3 = LayerNormalization([dim_model])

        self.dropout = nn.Dropout(self.config["dropout_p"])

    def forward(self, prev_preds: torch.Tensor, encoder_out: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        masked_attn_out = self.masked_attention(prev_preds, prev_preds, prev_preds, attn_mask)
        masked_attn_out = self.dropout(masked_attn_out)
        attn_in = self.layer_norm_1(prev_preds + masked_attn_out)

        attn_out = self.attention(encoder_out, attn_in, encoder_out)
        attn_out = self.dropout(attn_out)
        ffn_in = self.layer_norm_2(attn_in + attn_out)

        ffn_out = self.ffn(ffn_in)
        ffn_out = self.dropout(ffn_out)
        out = self.layer_norm_3(ffn_in + ffn_out)

        return out


class Transformer(nn.Module):

    def __init__(self, config: Dict):
        super(Transformer, self).__init__()
        self.config = config
        self.dim_model = config["dim_model"]
        max_seq_len = config["max_seq_len"]
        source_vocab_size = config["source_vocab_size"]
        target_vocab_size = config["target_vocab_size"]

        self.input_token_embed = nn.Embedding(source_vocab_size, self.dim_model)

        self.output_w = nn.Parameter(torch.randn((target_vocab_size, self.dim_model), dtype=torch.float32, requires_grad=True))
        self.output_b = nn.Parameter(torch.zeros((target_vocab_size), dtype=torch.float32, requires_grad=True))
        self.output_token_embed = nn.Embedding(target_vocab_size, self.dim_model, _weight=self.output_w)

        self.positional_encoding = PositionalEncoding(max_seq_len, self.dim_model, config["dropout_p"])

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(self.config["model_num_blocks"])])
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(self.config["model_num_blocks"])])

    def forward(self, input_tokens: torch.Tensor, prev_output_tokens: torch.Tensor, attn_mask: torch.Tensor):
        input = self.input_token_embed(input_tokens) * np.sqrt(self.dim_model)
        enc_x = self.positional_encoding(input)
        for encoder_block in self.encoder_blocks:
            enc_x = encoder_block(enc_x)

        prev_output = self.output_token_embed(prev_output_tokens) * np.sqrt(self.dim_model)
        dec_x = self.positional_encoding(prev_output)
        for decoder_block in self.decoder_blocks:
            dec_x = decoder_block(dec_x, enc_x, attn_mask)

        logits = torch.matmul(dec_x, self.output_w.T) + self.output_b

        return logits
