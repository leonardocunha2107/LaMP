import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer,DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from pdb import set_trace as stop 
from lamp import utils
from lamp.mask_handler import TrimHandler
import copy



class RNNDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        # dropout = 0

        super(RNNDecoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.n_tgt_vocab = n_tgt_vocab
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        self.attention_stack = nn.ModuleList([ScaledDotProductAttention(d_model,dropout=dropout)for _ in range(n_layers)])
        self.rnn_layer_stack = nn.ModuleList([
            nn.GRU(d_model+d_word_vec,d_model,batch_first=True,dropout=dropout)
            for _ in range(n_layers)])
        self.U = nn.Linear(self.d_model, self.n_tgt_vocab)
        self.V = nn.Linear(self.d_model, self.n_tgt_vocab)
        self.C = nn.Linear(self.d_model, self.n_tgt_vocab)

    def forward_step(self, input_var, decoder_hidden, encoder_outputs,dec_enc_attn_pad_mask=None):
        batch_size = input_var.size(0)
        embedded = self.tgt_word_emb(input_var)
        decoder_hidden = decoder_hidden.view(batch_size,1,-1)
        
        if encoder_outputs.size(1) == 1:
            dec_enc_attn_pad_mask=None

        for idx,dec_layer in enumerate(self.rnn_layer_stack):
            context,attn = self.attention_stack[idx](decoder_hidden.view(batch_size,1,-1),encoder_outputs,encoder_outputs,dec_enc_attn_pad_mask)
            rnn_input = torch.cat((embedded,context),2)
            embedded,decoder_hidden = dec_layer(rnn_input, decoder_hidden.view(1,batch_size,-1))

        output = self.U(decoder_hidden)
        output += self.V(embedded.view(batch_size,-1))
        output += self.C(context.view(batch_size,-1))

        return output, decoder_hidden, attn

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        batch_size = enc_output.size(0)
        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq,unsqueeze=False)
        dec_output = torch.zeros(tgt_seq.size(0),tgt_seq.size(1),self.n_tgt_vocab).cuda()
        dec_input = tgt_seq[:,0].unsqueeze(1)
        decoder_hidden = enc_output.mean(1)

        for di in range(tgt_seq.size(1)):
            decoder_output,decoder_hidden,step_attn=self.forward_step(dec_input,decoder_hidden,enc_output,dec_enc_attn_pad_mask)

            dec_output[:,di,:] = decoder_output
            dec_input = F.log_softmax(decoder_output.view(batch_size,-1),dim=1).topk(1)[1].view(batch_size,-1)

        return dec_output,


class MLPDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq_e, n_max_seq_d, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,enc_transform='mean'):
        super(MLPDecoder, self).__init__()
        self.n_max_seq = n_max_seq_e
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.enc_transform = enc_transform
        if enc_transform in ['flatten']: raise NotImplementedError
        self.linear1 = nn.Linear(d_model,d_model)
        self.linear4 = nn.Linear(d_model,n_tgt_vocab)

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        batch_size = src_seq.size(0)
        x = enc_output.float()

        out1 = self.dropout(F.relu(self.linear1(x)))
        dec_dropout = self.linear4(out1)
        return dec_dropout.view(batch_size,1,-1),
    

class GraphDecoder(nn.Module):
    def __init__(
            self,opt, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,n_head2=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,dropout2=0.1,
            no_dec_self_att=False,label_adj_matrix=None,label_mask=None,
            enc_vec=True,graph_conv=False,attn_type='softmax'):
        super(GraphDecoder, self).__init__()
        self.enc_vec = enc_vec
        self.dropout = nn.Dropout(dropout)
        self.constant_input = torch.arange(n_tgt_vocab).view(-1,1)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        
        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i,i] = 1 #This prevents Nan output in attention (otherwise 0 attn weights occurs)
            self.label_mask = utils.swap_0_1(label_adj_matrix,1,0).unsqueeze(0)
        else:
            if label_mask == 'inveye':
                self.label_mask = 1-torch.eye(n_tgt_vocab)
            elif label_mask == 'none':
                self.label_mask = None
            else:
                NotImplementedError
        
        self.layer_stack = nn.ModuleList()
        
        ##changes made to make model more dependendant on the encoding of each node
        self.mask_handler=None
        if opt.mask_handler=='trim':
            eps=opt.trim_eps if opt.trim_eps else 0.5
            crop_every=500 if not opt.crop_every else 500
            self.mask_handler=TrimHandler(n_tgt_vocab,crop_every=crop_every,eps=eps)
        if opt.less_attn:
            self.layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,
                                                 dropout2=dropout2,no_dec_self_att=no_dec_self_att,attn_type=attn_type))           
            for _ in range(n_layers-1):
                            self.layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,
                                                                 dropout2=dropout2,no_dec_self_att=no_dec_self_att,attn_type=attn_type,no_enc_attn=True))           

        else:
            for _ in range(n_layers):
                self.layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, 
                                                     dropout=dropout,dropout2=dropout2,no_dec_self_att=no_dec_self_att,attn_type=attn_type))           


    def forward(self, tgt, src_seq, enc_output,return_attns=False, int_preds=False):
        batch_size = src_seq.size(0)
        if int_preds: int_outs = []
        if return_attns: dec_slf_attns, dec_enc_attns = [], []

        tgt_seq = self.constant_input.repeat(1,batch_size).transpose(0,1).cuda()

        dec_input = self.tgt_word_emb(tgt_seq)

        dec_enc_attn_pad_mask = None
        if not self.enc_vec:
            dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq[:,0:enc_output.size(1)])
        
        ###changes mostly here on defining attn_mask
        if self.mask_handler:
            dec_slf_attn_mask=self.mask_handler.get_mask(batch_size)
        elif self.label_mask is not None:
            dec_slf_attn_mask = self.label_mask.repeat(batch_size,1,1).cuda().byte()
        else:
            dec_slf_attn_mask = None
            
        dec_output = dec_input
        for idx,dec_layer in enumerate(self.layer_stack):
            dec_output, dec_output_int, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_pad_mask)
            
            if self.mask_handler and idx==1 and  dec_slf_attn:
                self.mask_handler.push(dec_slf_attn)
                
            
            
            
            
            if int_preds:
                if dec_output_int is not None:
                    int_outs += [dec_output_int]
                int_outs += [dec_output]

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if int_preds:
            return dec_output, int_outs
        elif return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output, None         
