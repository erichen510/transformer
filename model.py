import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer

from torch.autograd import Function


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        '''

        :param n_src_vocab: 词汇量
        :param len_max_seq: 最长串
        :param d_word_vec:
        :param d_model: 512
        :param d_inner:PositionwiseFeedForward一维卷积的中间 channel
        :param dropout:
        '''

        super().__init__()

        n_position = len_max_seq + 1

        '''
        对进来的Sequence进行wording embedding 和 position encondung 
        '''
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_model, padding_idx=Constants.PAD)
        #todo:将BERT的position加进来替换掉现在的
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = [] #把attention放入list

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq) #没有词的地方补0
        non_pad_mask = get_non_pad_mask(src_seq)

        #embedding的结果和位置结果相加作为encoder的输入
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,



class Transformer(nn.Module):

    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.linear = nn.Linear(d_model, 768, bias=False)

    def forward(self, src_seq, src_pos):

        enc_output, *_ = self.encoder(src_seq, src_pos)

        output = self.linear(enc_output)

        return output


class GradientRerverse(Function):
    '''
    forward 是 identity
    backward 是 乘一个负值的梯度
    '''

    @staticmethod
    def forward(ctx, input, lambda_):
        '''

        :param ctx: 类似于SELF的作用
        :param input:
        :param lambda_:
        :return:
        '''
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GRL(nn.Module):
      def __init__(self, lambda_ =1):
          super(GRL, self).__init__()
          self.lambda_ = lambda_

      def forward(self, x):
          return GradientRerverse(x, self.lambda_)


class generate_model(nn.Module):
    def __init__(self):
        super(generate_model, self).__init__()
        left_class = 2
        right_class = 31
        self.transformer_left = Transformer(n_src_vocab=100, len_max_seq=20)
        self.left_socre = nn.Linear(768, left_class)
        self.transformer_right = Transformer(n_src_vocab=100, len_max_seq=20)
        self.right_score = nn.Linear(768, right_class)
        self.GRL = GRL()

    def forward(self, src_seq, src_pos):
        out_left = self.transformer_left(src_seq, src_pos)
        out_left = self.left_socre(out_left)
        out_right = self.transformer_right(src_seq, src_pos)
        out_right = self.GRL(out_right)
        out_right = self.right_score(out_right)

        return  out_left, out_right

