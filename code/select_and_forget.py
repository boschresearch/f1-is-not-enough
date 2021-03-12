""" F1 is Not Enough! Models and Evaluation Towards User-Centered Explainable Question Answering (EMNLP 2020).
Copyright (c) 2021 Robert Bosch GmbH
@author: Hendrik Schuff
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# This file is based on https://github.com/qipeng/golden-retriever/ which in turn is based on https://github.com/hotpotqa/hotpot.

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

# We reuse the modules implemented by Qi et al.
from .common_modules import LockedDropout, EncoderRNN, BiAttention


class SelectAndForgetModel(nn.Module):
    def __init__(self, word_mat, char_mat, glove_dim, char_dim, char_hidden, hidden, keep_prob):
        super().__init__()
        # Joint encoding
        self.word_dim = glove_dim
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False
        self.dropout = LockedDropout(1 - keep_prob)
        self.char_hidden = char_hidden
        self.hidden = hidden

        # Answer encoding
        self.char_emb_ans = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb_ans.weight.data.copy_(torch.from_numpy(char_mat))
        self.char_cnn_ans = nn.Conv1d(char_dim, char_hidden, 5)
        self.rnn_ans = EncoderRNN(self.word_dim + self.char_hidden + 1, hidden, 1, True, True, 1 - keep_prob, False)

        # Supporting facts encoding
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))
        self.char_cnn = nn.Conv1d(char_dim, char_hidden, 5)
        self.rnn = EncoderRNN(self.word_dim + self.char_hidden + 1, hidden, 1, True, True, 1 - keep_prob, False)

        # Answer prediction
        self.qc_att = BiAttention(hidden * 2, 1 - keep_prob)
        self.linear_1 = nn.Sequential(
            nn.Linear(hidden * 6, hidden * 2),
            nn.Tanh()
        )
        self.rnn_2 = EncoderRNN(hidden * 2, hidden, 1, False, True, 1 - keep_prob, False)
        self.self_att = BiAttention(hidden * 2, 1 - keep_prob)
        self.linear_2 = nn.Sequential(
            nn.Linear(hidden * 6, hidden * 2),
            nn.Tanh()
        )
        self.rnn_sp = EncoderRNN(hidden * 2, hidden, 1, False, True, 1 - keep_prob, False)
        self.linear_sp = nn.Linear(hidden * 2, 1)
        self.rnn_start = EncoderRNN(hidden * 4, hidden, 1, False, True, 1 - keep_prob, False)
        self.linear_start = nn.Linear(hidden * 2, 1)
        self.rnn_end = EncoderRNN(hidden * 4, hidden, 1, False, True, 1 - keep_prob, False)
        self.linear_end = nn.Linear(hidden * 2, 1)
        self.rnn_type = EncoderRNN(hidden * 4, hidden, 1, False, True, 1 - keep_prob, False)
        self.linear_type = nn.Linear(hidden * 2, 3)
        self.cache_S = 0

        # Supporting facts predictions
        self.qc_att_spp = BiAttention(hidden * 2, 1 - keep_prob)
        self.linear_1_spp = nn.Sequential(
            nn.Linear(hidden * 6, hidden * 2),
            nn.Tanh()
        )
        self.rnn_2_spp = EncoderRNN(hidden * 2, hidden, 1, False, True, 1 - keep_prob, False)
        self.self_att_spp = BiAttention(hidden * 2, 1 - keep_prob)
        self.linear_2_spp = nn.Sequential(
            nn.Linear(hidden * 6, hidden * 2),
            nn.Tanh()
        )
        self.rnn_sp_spp = EncoderRNN(hidden * 2, hidden, 1, False, True, 1 - keep_prob, False)
        self.linear_sp_spp = nn.Linear(hidden * 2, 1)

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def rnn_over_context(self, rnn, x, lens):
        batch_size, num_of_paragraphs, para_len, hidden_dim = x.size()
        x = self.dropout(x.view(batch_size, num_of_paragraphs * para_len, hidden_dim))
        x = x.view(batch_size * num_of_paragraphs, para_len, hidden_dim)
        lens = lens.view(-1)
        l1 = torch.max(lens, lens.new_ones(1))
        y = rnn(x, l1)
        return y.masked_fill((lens == 0).unsqueeze(1).unsqueeze(2), 0).view(batch_size, num_of_paragraphs, para_len, -1)

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping,
                end_mapping, all_mapping, max_len, return_yp=False):
        '''
        Dimensions:
        
            b      = batch_size
            l_con  = #tokens in context  (max per batch, padded)
            l_ques = #tokens in question (max per batch, padded)
            l_f    = #facts (paragraphs) over all articles
            n_art  = #articles per query
            m_par  = maximum #tokens per paragraph (including title)
            m_char = maximum #characters per word
            e_char = size of character embedding, h = size of hidden layers, e_word = size of word embedding,
                     h_cnn = #output channels of the character CNN
        
            context_idxs             b x n_art x m_par                     
            ques_idx                 b x l_ques
            context_char_idxs        b x n_art x m_par x m_char       
            ques_char_idxs           b x l_ques x m_char       
            context_lens             b x n_art       
            start_mapping            b x l_f x (n_art * m_par)
            end_mapping                     - " -     
            all_mapping                     - " -
            max_len                  = l_con \in R      
        '''

        # Joint encoding
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(
            -1), context_idxs.size(0)  # n_art, l_ques, m_char, b

        batch_size, num_of_paragraphs, para_len = context_idxs.size()  # b, n_art, m_par
        context_idxs = context_idxs.reshape(-1, para_len)  # (b * n_art) x m_par
        context_mask = (context_idxs > 0).float()  # (b * n_art) x m_par
        ques_mask = (ques_idxs > 0).float()  # b x l_ques

        context_word = self.word_emb(context_idxs)  # (b * n_art) x m_par x e_word
        ques_word = self.word_emb(ques_idxs)  # b * l_ques x e_word

        # Answer encoding
        context_ch_ans = self.char_emb_ans(context_char_idxs)  # b x n_art x m_par x m_char x e_char
        ques_ch_ans = self.char_emb_ans(ques_char_idxs)  # b x l_ques x m_char x e_char
        context_ch_ans = self.char_cnn_ans(
            context_ch_ans.view(batch_size * num_of_paragraphs * para_len, char_size, -1).permute(0, 2,
                                                                                                  1).contiguous()).max(
            dim=-1)[0].view(batch_size * num_of_paragraphs, para_len, -1)  # (b * n_art) x m_par x h_cnn
        ques_ch_ans = \
            self.char_cnn_ans(
                ques_ch_ans.view(batch_size * ques_size, char_size, -1).permute(0, 2, 1).contiguous()).max(
                dim=-1)[0].view(bsz, ques_size, -1)  # b x l_ques x h_cnn

        context_output_ans = torch.cat(
            [context_word, context_ch_ans, context_word.new_zeros((context_word.size(0), context_word.size(1), 1))],
            dim=2).view(batch_size, num_of_paragraphs, para_len, -1)  # b x n_art x m_par x (e_word + c_cnn + 1)
        ques_output_ans = torch.cat(
            [ques_word, ques_ch_ans, ques_word.new_ones((ques_word.size(0), ques_word.size(1), 1))],
            dim=2)  # b x l_ques x (e_word + c_cnn + 1)

        context_output_ans = self.rnn_over_context(self.rnn_ans, context_output_ans,
                                                   context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        ques_output_ans = self.rnn_ans(self.dropout(ques_output_ans))  # b x l_ques x (2*h) (bidirectional)

        # Supporting facts encoding
        context_ch_spp = self.char_emb(context_char_idxs)  # b x n_art x m_par x m_char x e_char
        ques_ch_spp = self.char_emb(ques_char_idxs)  # b x l_ques x m_char x e_char

        context_ch_spp = self.char_cnn(
            context_ch_spp.view(batch_size * num_of_paragraphs * para_len, char_size, -1).permute(0, 2,
                                                                                                  1).contiguous()).max(
            dim=-1)[0].view(batch_size * num_of_paragraphs, para_len, -1)  # (b * n_art) x m_par x h_cnn
        ques_ch_spp = \
            self.char_cnn(ques_ch_spp.view(batch_size * ques_size, char_size, -1).permute(0, 2, 1).contiguous()).max(
                dim=-1)[0].view(bsz, ques_size, -1)  # b x l_ques x h_cnn

        context_output_spp = torch.cat(
            [context_word, context_ch_spp, context_word.new_zeros((context_word.size(0), context_word.size(1), 1))],
            dim=2).view(batch_size, num_of_paragraphs, para_len, -1)  # b x n_art x m_par x (e_word + c_cnn + 1)
        ques_output_spp = torch.cat(
            [ques_word, ques_ch_spp, ques_word.new_ones((ques_word.size(0), ques_word.size(1), 1))],
            dim=2)  # b x l_ques x (e_word + c_cnn + 1)

        context_output_spp = self.rnn_over_context(self.rnn, context_output_spp,
                                                   context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        ques_output_spp = self.rnn(self.dropout(ques_output_spp))  # b x l_ques x (2*h) (bidirectional)

        # Supporting facts prediction
        qc_hid_spp = torch.cat([context_output_spp.view(batch_size, num_of_paragraphs * para_len, -1), ques_output_spp],
                               1)  # b x (n_art * m_par + l_ques) x 2*h
        qc_mask_spp = torch.cat([context_mask.view(batch_size, num_of_paragraphs * para_len), ques_mask],
                                1)  # b x (n_art * m_art + l_ques)
        output_spp = self.qc_att_spp(qc_hid_spp, qc_hid_spp, qc_mask_spp,
                                     qc_mask_spp)  # b x (n_art * m_par + l_ques) x (3*2*h)
        output_spp = self.linear_1_spp(self.dropout(output_spp))  # b x (n_art * m_par + l_ques) x (2*h)
        c_output_spp = output_spp[:,
                       :num_of_paragraphs * para_len].contiguous()  # b x (n_art * m_par) x (2*h) (context part)
        q_output_spp = output_spp[:,
                       num_of_paragraphs * para_len:].contiguous()  # b x l_ques x (2*h)          (question part)
        output_t_spp = self.rnn_over_context(self.rnn_2_spp,
                                             c_output_spp.view(batch_size, num_of_paragraphs, para_len, -1),
                                             context_lens)  # b x n_art x m_par x (2*h)
        ques_output2_spp = self.rnn_2_spp(self.dropout(q_output_spp))  # b x l_ques x (2*h) (bidirectional)
        qc_hid2_spp = torch.cat([output_t_spp.view(batch_size, num_of_paragraphs * para_len, -1), ques_output2_spp],
                                1)  # b x (n_art * m_par + l_ques) x (2*h)
        output_t_spp = self.self_att_spp(qc_hid2_spp, qc_hid2_spp, qc_mask_spp,
                                         qc_mask_spp)  # b x (n_art * m_par + l_ques) x (3*2*h)
        output_t_spp = self.linear_2_spp(self.dropout(output_t_spp))  # b x (n_art * m_par + l_ques) x (2*h)
        output_spp = output_spp + output_t_spp  # b x (n_art * m_par + l_ques) x (2*h)
        output_spp = output_spp[:,
                     :num_of_paragraphs * para_len].contiguous()  # discard question output  b x (n_art * m_par) x (2*h)
        output_spp = output_spp.view(batch_size, num_of_paragraphs, para_len, -1)  # b x  n_art x m_par x (2*h)
        sp_output_spp = self.rnn_over_context(self.rnn_sp_spp, output_spp,
                                              context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        sp_output_spp = sp_output_spp.view(batch_size, num_of_paragraphs * para_len, -1)  # b x (n_art * m_par) x (2*h)
        sp_output_spp = torch.matmul(all_mapping, sp_output_spp) / (
                all_mapping.float().sum(-1, keepdim=True) + 1e-6)  # b x l_f x (2*h)
        sp_output_t_spp = self.linear_sp_spp(self.dropout(sp_output_spp))  # b x l_f x 1
        sp_output_aux_spp = sp_output_t_spp.new_zeros(sp_output_t_spp.size(0), sp_output_t_spp.size(1),
                                                      1)  # b x l_f x 1
        predict_support_spp = torch.cat([sp_output_aux_spp, sp_output_t_spp], dim=-1).contiguous()  # b x l_f x 2

        # "Forgetting" facts that are not predicted to be relevant

        b = bsz
        n_art = num_of_paragraphs
        m_par = para_len
        two_h = 2 * self.hidden
        l_f = start_mapping.size(1)
        relevance_logits = predict_support_spp
        predict_support = relevance_logits  # b x l_f x 2
        # Same procedure as in prediction decoding
        thresholded_rvs_pos = (torch.sigmoid(relevance_logits[:, :, 1] - relevance_logits[:, :, 0]) > 0.33).float()
        thresholded_rvs_neg = [thresholded_rvs_pos == 0.0][0].float()
        relevance_rvs = torch.cat([thresholded_rvs_neg.view(b, l_f, 1), thresholded_rvs_pos.view(b, l_f, 1)],
                                  dim=-1)  # b x l_f x 2
        # We arbitrarily define that if a value in the SECOND column is one,
        # this means the fact is predicted to be relevant.
        relevance_rvs_rel = relevance_rvs[:, :, 1].view(b, l_f, 1)  # b x l_f x 1
        # SP prediction
        sp_predictions_single = relevance_rvs_rel.view(b, l_f, 1)  # b x l_f x 1
        # Map facts to tokens
        token_mask = (all_mapping.transpose(1, 2) * sp_predictions_single.transpose(1, 2))  # b x (n_art * m_par) x l_f
        token_mask = token_mask.max(dim=-1)[0].view(b, n_art * m_par, 1)  # b x (n_art * m_par) x 1
        masked_contexts = context_output_ans.view(b, n_art * m_par, two_h) * token_mask  # b x (n_art * m_par) x (2*h)
        # Reshape into original format
        masked_contexts = masked_contexts.view(b, n_art, m_par, two_h)  # b x n_art x m_par x (2*h)
        # Now, we replace the context_output with the masked version
        context_output_masked = masked_contexts  # b x n_art x m_par x (2*h)

        # Answer prediction
        qc_hid = torch.cat([context_output_masked.view(batch_size, num_of_paragraphs * para_len, -1), ques_output_ans],
                           1)  # b x (n_art * m_par + l_ques) x 2*h
        qc_mask = torch.cat([context_mask.view(batch_size, num_of_paragraphs * para_len), ques_mask],
                            1)  # b x (n_art * m_art + l_ques)
        output = self.qc_att(qc_hid, qc_hid, qc_mask, qc_mask)  # b x (n_art * m_par + l_ques) x (3*2*h)
        output = self.linear_1(self.dropout(output))  # b x (n_art * m_par + l_ques) x (2*h)
        c_output = output[:,
                   :num_of_paragraphs * para_len].contiguous()  # b x (n_art * m_par) x (2*h) (extract the context part)
        q_output = output[:,
                   num_of_paragraphs * para_len:].contiguous()  # b x l_ques x (2*h) (extract the question part)
        output_t = self.rnn_over_context(self.rnn_2, c_output.view(batch_size, num_of_paragraphs, para_len, -1),
                                         context_lens)  # b x n_art x m_par x (2*h)
        ques_output2 = self.rnn_2(self.dropout(q_output))  # b x l_ques x (2*h) (bidirectional)
        qc_hid2 = torch.cat([output_t.view(batch_size, num_of_paragraphs * para_len, -1), ques_output2],
                            1)  # b x (n_art * m_par + l_ques) x (2*h)
        output_t = self.self_att(qc_hid2, qc_hid2, qc_mask, qc_mask)  # b x (n_art * m_par + l_ques) x (3*2*h)
        output_t = self.linear_2(self.dropout(output_t))  # b x (n_art * m_par + l_ques) x (2*h)
        output = output + output_t  # b x (n_art * m_par + l_ques) x (2*h)
        output = output[:,
                 :num_of_paragraphs * para_len].contiguous()  # discard question output  b x (n_art * m_par) x (2*h)
        output = output.view(batch_size, num_of_paragraphs, para_len, -1)  # b x  n_art x m_par x (2*h)
        sp_output = self.rnn_over_context(self.rnn_sp, output,
                                          context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        sp_output = sp_output.view(batch_size, num_of_paragraphs * para_len, -1)  # b x (n_art * m_par) x (2*h)
        sp_output = torch.matmul(all_mapping, sp_output) / (
                all_mapping.float().sum(-1, keepdim=True) + 1e-6)  # b x l_f x (2*h)
        sp_output = torch.matmul(all_mapping.transpose(1, 2), sp_output)  # b x (n_art * m_par) x (2*h)
        output_start = torch.cat([output, sp_output.view(batch_size, num_of_paragraphs, para_len, -1)],
                                 dim=-1)  # b x n_art x m_par x (2*2*h)
        output_start = self.rnn_over_context(self.rnn_start, output_start,
                                             context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        output_end = torch.cat([output, output_start], dim=-1)  # b x n_art x m_par x (2*2*h)
        output_end = self.rnn_over_context(self.rnn_end, output_end,
                                           context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        output_type = torch.cat([output, output_end], dim=-1)  # b x n_art x m_par x (2*2*h)
        output_type = self.rnn_over_context(self.rnn_type, output_type,
                                            context_lens)  # b x n_art x m_par x (2*h) (bidirectional)
        predict_start = self.linear_start(
            self.dropout(output_start.view(batch_size, num_of_paragraphs * para_len, -1))).view(batch_size,
                                                                                                num_of_paragraphs,
                                                                                                para_len)  # b x n_art x m_par
        predict_end = self.linear_end(self.dropout(output_end.view(batch_size, num_of_paragraphs * para_len, -1))).view(
            batch_size, num_of_paragraphs, para_len)  # b x n_art x m_par
        output_type = output_type.view(batch_size, num_of_paragraphs, para_len,
                                       output_type.size(-1))  # b x n_art x m_par x (2*h)

        # dissect padded sequences of each paragraph and make padded sequence for each example
        # as predictions so we don't have to mess with the data format
        cumlens = context_lens.sum(1)  # b
        logit1 = []
        logit2 = []
        p0_type = []
        for i in range(context_lens.size(0)):
            logit1.append(torch.cat([predict_start[i, j, :context_lens[i][j]] for j in range(context_lens.size(1))] + [
                predict_start.new_full((max_len - cumlens[i],), -1e30)], dim=0))
            logit2.append(torch.cat([predict_end[i, j, :context_lens[i][j]] for j in range(context_lens.size(1))] + [
                predict_end.new_full((max_len - cumlens[i],), -1e30)], dim=0))
            p0_type.append(torch.cat([output_type[i, j, :context_lens[i][j]] for j in range(context_lens.size(1))] + [
                predict_end.new_full((max_len - cumlens[i], output_type.size(-1)), -1e30)], dim=0))
        logit1 = torch.stack(logit1)  # b x l_con
        logit2 = torch.stack(logit2)  # b x l_con
        p0_type = torch.stack(p0_type)  # b x l_con x (2*h)
        prediction_type = self.linear_type(self.dropout(p0_type).max(1)[0])  # b x 3

        if not return_yp: return logit1, logit2, prediction_type, predict_support

        outer = logit1[:, :, None] + logit2[:, None]  # b x l_con x l_con
        outer_mask = self.get_output_mask(outer)  # l_con x l_con
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))  # b x l_con x l_con
        yp = outer.view(outer.size(0), -1).max(1)[1]  # b
        yp1 = yp // outer.size(1)  # b
        yp2 = yp % outer.size(1)  # b
        return logit1, logit2, prediction_type, predict_support, yp1, yp2
