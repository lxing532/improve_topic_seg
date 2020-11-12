from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import maybe_cuda, setup_logger, unsort
import numpy as np
from times_profiler import profiler


logger = setup_logger(__name__, 'train.log')
profilerLogger = setup_logger("profilerLogger", 'profiler.log', True)

def no_name(X, i, win_size, hidden, self_attn):

    X1 = X.unsqueeze(0)
    Y1 = X.unsqueeze(1)
    X2 = X1.repeat(X.shape[0],1,1)
    Y2 = Y1.repeat(1,X.shape[0],1)

    output = 0

    Z = torch.cat([X2,Y2],-1)
    if i <= win_size:
        a = Z[i,:,0:int(Z.size()[-1]/2)] 
        a_norm = a / a.norm(dim=1)[:, None]
        b = Z[i,:,int(Z.size()[-1]/2):]
        b_norm = b / b.norm(dim=1)[:, None]
        z = torch.cat([Z[i,:],F.sigmoid(torch.diag(torch.mm(a_norm,b_norm.transpose(0,1)))).unsqueeze(-1)],-1)
        attn_weight = F.softmax(self_attn(z), dim=0).permute(1,0)

        output = attn_weight.matmul(Z[i,:,0:int(Z.size()[-1]/2)])
        
        #torch.stack([F.softmax(attn_weight[i, :lengths[i]], dim=0).matmul(padded_output[i, :lengths[i]]) for i in range(len(lengths))], dim=0)
    else:
        a = Z[win_size,:,0:int(Z.size()[-1]/2)] 
        a_norm = a / a.norm(dim=1)[:, None]
        b = Z[win_size,:,int(Z.size()[-1]/2):]
        b_norm = b / b.norm(dim=1)[:, None]
        z = torch.cat([Z[win_size,:],F.sigmoid(torch.diag(torch.mm(a_norm,b_norm.transpose(0,1)))).unsqueeze(-1)],-1)
        attn_weight = F.softmax(self_attn(z), dim=0).permute(1,0)

        output = attn_weight.matmul(Z[win_size,:,0:int(Z.size()[-1]/2)])

    return output


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden))), \
           Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden)))


class SentenceEncodingRNN(nn.Module):
    def __init__(self, input_size, hidden, num_layers):
        super(SentenceEncodingRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True)

        self.mlp = nn.Sequential(nn.Linear(2*hidden, 2*hidden), nn.Tanh())
        self.context_vector = nn.Parameter(torch.Tensor(2*hidden))
        self.context_vector.data.normal_(0, 0.1)

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        s = zero_state(self, batch_size)
        packed_output, _ = self.lstm(x, s)
        padded_output, lengths = pad_packed_sequence(packed_output) # (max sentence len, batch, 256) 
        
        # attention
        padded_output = padded_output.permute(1,0,2)
        word_annotation = self.mlp(padded_output)
        attn_weight = word_annotation.matmul(self.context_vector)
        attended_outputs = torch.stack([F.softmax(attn_weight[i, :lengths[i]], dim=0).matmul(padded_output[i, :lengths[i]]) for i in range(len(lengths))], dim=0)

        return attended_outputs
        


class Model(nn.Module):
    def __init__(self, sentence_encoder, hidden=128, num_layers=2):
        super(Model, self).__init__()

        self.sentence_encoder = sentence_encoder

        self.sentence_lstm = nn.LSTM(input_size=768+2*hidden, # The dimension of BERT embedding + output of BiLSTM hidden state
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        self.sentence_lstm_2 = nn.LSTM(input_size=sentence_encoder.hidden * 4,
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        # We have two labels
        self.h2s = nn.Linear(hidden * 2, 2)

        self.num_layers = num_layers
        self.hidden = hidden

        self.criterion = nn.CrossEntropyLoss()
        module_fwd = [nn.Linear(hidden, hidden), nn.Tanh()]
        self.fwd = nn.Sequential(*module_fwd)

        modules = [nn.Linear(4*hidden+1, 1)]
        self.self_attn = nn.Sequential(*modules)


    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = Variable(maybe_cuda(s.unsqueeze(0).unsqueeze(0)))
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)


    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0,0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def max_pooling_similarity_computing(self, doc_output, segment_idx):
        similarities = Variable(maybe_cuda(torch.Tensor([])))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #ln = nn.LayerNorm(512, elementwise_affine=False)
        seg_outputs = []
        index = 0
        doc_output = F.softmax(doc_output)
        for i, idx in enumerate(segment_idx):
            if i == 0:
                seg_output = doc_output[0 : segment_idx[i]+1, :]
            elif i == len(segment_idx)-1:
                seg_output = doc_output[segment_idx[i-1]+1 : , :]
            else:
                seg_output = doc_output[segment_idx[i-1]+1:segment_idx[i]+1, :]
            seg_outputs.append(seg_output)

        maxes = Variable(maybe_cuda(torch.zeros(len(seg_outputs), self.hidden * 2)))
        for i in range(len(seg_outputs)):
            maxes[i, :] = torch.max(seg_outputs[i], 0)[0]
            #maxes[i, :] = torch.mean(seg_outputs[i], 0)
        if len(seg_outputs) > 1:
            tensor_1 = maxes[:-1, :]
            tensor_2 = maxes[1:, :]
            similarities = cos(tensor_1, tensor_2)
            #similarity += (tensor_1 * tensor_2).sum()/(tensor_1.size()[0])
            #similarities = torch.diag(torch.mm(tensor_1, tensor_2.permute(1,0)))
        return similarities

    def similarity_computing_inner(self, doc_output, segment_idx):
        similarities = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        seg_outputs = []
        index = 0
        doc_output = F.softmax(doc_output)
        for i, idx in enumerate(segment_idx):
            if i == 0:
                seg_output = doc_output[0 : segment_idx[i]+1, :]
            elif i == len(segment_idx)-1:
                seg_output = doc_output[segment_idx[i-1]+1 : , :]
            else:
                seg_output = doc_output[segment_idx[i-1]+1:segment_idx[i]+1, :]
            seg_outputs.append(seg_output)

        for i in range(len(seg_outputs)):
            sent_idx = maybe_cuda(torch.LongTensor([k for k in range(seg_outputs[i].size()[0])]))
            if seg_outputs[i].size()[0] > 1:
                pairs = torch.combinations(sent_idx)
                pair_sims = []
                for p in pairs:
                    pair_sims.append(cos(seg_outputs[i][p[0],:].unsqueeze(0),seg_outputs[i][p[1],:].unsqueeze(0)))
                similarities.append(sum(pair_sims)/len(pair_sims))
            else:
                continue

        return Variable(maybe_cuda(torch.Tensor(similarities)))

    def forward(self, batch, sent_bert_vec, target_idx):
        batch_size = len(batch)

        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))

        lengths = [s.size()[0] for s in all_batch_sentences]
        sort_order = np.argsort(lengths)[::-1]
        sorted_sentences = [all_batch_sentences[i] for i in sort_order]
        sorted_lengths = [s.size()[0] for s in sorted_sentences]

        max_length = max(lengths)

        padded_sentences = [self.pad(s, max_length) for s in sorted_sentences]
        big_tensor = torch.cat(padded_sentences, 1)  # (max_length, batch size, 300)
        packed_tensor = pack_padded_sequence(big_tensor, sorted_lengths)
        encoded_sentences = self.sentence_encoder(packed_tensor)
        unsort_order = Variable(maybe_cuda(torch.LongTensor(unsort(sort_order))))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        sent_bert_vec_conc = sent_bert_vec[0]
        for i in range(1,len(sent_bert_vec)):
            sent_bert_vec_conc = torch.cat((sent_bert_vec_conc, sent_bert_vec[i]),0)
        #unsorted_encodings = torch.autograd.Variable(maybe_cuda(sent_bert_vec_conc), requires_grad=True)
        sent_bert_vec_conc = torch.autograd.Variable(maybe_cuda(sent_bert_vec_conc), requires_grad=True)
        unsorted_encodings = torch.cat((unsorted_encodings, sent_bert_vec_conc),1)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index : end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=batch_size))
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        # computing the consecutive sentence pairs' similarities...
        window = 1
        doc_outputs = []; sims_outputs = [];
        pd_f = (0,0,window,0); pd_b = (0,0,0,window)
        for i, doc_len in enumerate(ordered_doc_sizes):
            #doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # *** -1 to remove last prediction ***
            batch_x = padded_x[0:doc_len, i, :]

            forward_padded_x = self.fwd(batch_x[:-1, :self.hidden] - F.pad(batch_x[:-1, :self.hidden], pd_f, 'constant', 0)[:-window,:])
            backward_padded_x = self.fwd(batch_x[1:, self.hidden:] - F.pad(batch_x[1:, self.hidden:], pd_b, 'constant', 0)[window:,:]).permute(1,0)  
            sims_outputs.append(F.sigmoid(torch.diag(torch.mm(forward_padded_x, backward_padded_x))))


        doc_outputs = []; doc_outputs_complete = [];
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction
            doc_outputs_complete.append(padded_x[0:doc_len, i, :])

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        unsorted_sims_outputs = [sims_outputs[i] for i in unsort(ordered_document_idx)]
        unsorted_doc_outputs_complete = [doc_outputs_complete[i] for i in unsort(ordered_document_idx)]

        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)
        sims_outputs = torch.cat(unsorted_sims_outputs, 0).unsqueeze(1)
        
        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len, i, :])
        
        win_size = 3 ; encoded_documents_2 = [];
        for doc_output in doc_outputs:
            doc_l = doc_output.size()[0]
            new_one = []
            for i in range(doc_l):
                if i-win_size < 0:
                    new_one.append(no_name(doc_output[0:i+win_size+1],i, win_size, self.hidden, self.self_attn))
                else:
                    new_one.append(no_name(doc_output[i-win_size:i+win_size+1],i, win_size, self.hidden, self.self_attn))

            encoded_documents_2.append(torch.stack(new_one, dim = 0).squeeze(1))

        encoded_documents_2 = [torch.cat([doc_outputs[i],encoded_documents_2[i]], -1) for i in range(len(doc_outputs))]
        doc_outputs = encoded_documents_2

        padded_docs = [self.pad_document(d, max_doc_size) for d in doc_outputs]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)

        sentence_lstm_output, _ = self.sentence_lstm_2(packed_docs, zero_state(self, batch_size=batch_size)) #till now, no sentence is removed
        #sentence_lstm_output = packed_docs
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        doc_outputs = [];
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len-1, i, :]) 

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)
        
        x = self.h2s(sentence_outputs)

        #return x, all_similarities.mean()
        return x, sims_outputs


def create():
    sentence_encoder = SentenceEncodingRNN(input_size=300,
                                           hidden=256,
                                           num_layers=2)
    return Model(sentence_encoder, hidden=256, num_layers=2)