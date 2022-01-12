import os
import sys
import traceback
import types

import torch
import torch.nn as nn
import torch.optim as optim

from diora.net.diora import DioraTreeLSTM
from diora.net.diora import DioraMLP
from diora.net.diora import DioraMLPShared

from diora.logging.configuration import get_logger

import transformers

import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    name = 'reconstruct_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        input_size = self.embeddings.weight.shape[1]
        size = diora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples)

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('ec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        lossfn = nn.MultiMarginLoss(margin=self.margin)
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        self.loss_hook(sentences, neg_samples, inputs)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_loss=loss)

        return loss, ret


class ReconstructionSoftmaxLoss(nn.Module):
    name = 'reconstruct_softmax_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionSoftmaxLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin
        self.input_size = input_size

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, neg_samples, diora, info):
        # print('sentences shape: ', sentences.shape)
        # print('sentence 1: ', sentences[0])
        # print('neg_samples: ', neg_samples.shape)
        # print('outside_h.shape: ', diora.outside_h.shape)
        batch_size, length = sentences.shape
        input_size = self.input_size
        size = diora.outside_h.shape[-1]
        k = self.k_neg
        # print('k: ', k)

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples.unsqueeze(0))

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        # print('emb_neg: ', emb_neg.shape)
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))
        # print('proj_neg: ', proj_neg.shape)
        
        ## The score.
        # print('proj_pos: ', proj_pos.shape)
        # print('proj_neg: ', proj_neg.shape)
        # print('cell: ', cell.shape)
        # proj_pos:  torch.Size([32, 11, 400])
        # proj_neg:  torch.Size([1, 100, 400])
        # cell:  torch.Size([32, 11, 1, 400])
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('zec,abxc->abe', proj_neg, cell)
        # xp:  torch.Size([32, 11, 1])
        # xn:  torch.Size([32, 11, 100])
        # print('xp: ', xp.shape)
        # print('xn: ', xn.shape)
        score = torch.cat([xp, xn], 2)

        # print('score: ', score.shape)
        # score shape: torch.Size([32, 11, 101])

        # Calculate loss.
        lossfn = nn.CrossEntropyLoss()
        # (batch_size x )
        inputs = score.view(batch_size * length, k + 1)
        # print('inputs: ', inputs)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)
        #
        self.loss_hook(sentences, neg_samples, inputs)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_softmax_loss=loss)
        return loss, ret


class ReconstructionBERTSoftmaxLoss(nn.Module):
    name = 'reconstruct_bert_softmax_loss'

    def __init__(self, vocab_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionBERTSoftmaxLoss, self).__init__()
        self.vocab_size = vocab_size

        self.cls = nn.Parameter(torch.FloatTensor(vocab_size, size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parametervocab_sizes(self):
        self.cls.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, batch, neg_samples, diora, info):
        # sentences is the label here
        
        # print('sentences shape: ', sentences.shape)
        # print('sentence 1: ', sentences[0])
        # print('neg_samples: ', neg_samples.shape)
        # print('outside_h.shape: ', diora.outside_h.shape)
        labels = batch['origin_input_ids']
        batch_size, length = labels.shape
        vocab_size = self.vocab_size
        size = diora.outside_h.shape[-1]
        
        # Calculate scores.
        cell = diora.outside_h[:, :length].view(batch_size, length, -1)

        ## The projected samples.
        score = torch.matmul(cell, torch.t(self.cls))
        
        ## The score.
        # proj_pos:  torch.Size([32, 11, 400])
        # proj_neg:  torch.Size([1, 100, 400])
        # cell:  torch.Size([32, 11, 1, 400])
        # xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        # xp:  torch.Size([32, 11, 1])
        # xn:  torch.Size([32, 11, 100])
        
        # score = torch.cat([xp, xn], 2)
        # score shape: torch.Size([32, 11, 101])

        # Calculate loss.
        lossfn = nn.CrossEntropyLoss()
        # (batch_size x )
        inputs = score.view(batch_size * length, vocab_size)
        # print('inputs: ', inputs)
        device = torch.cuda.current_device() if self._cuda else None
        # outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)
        outputs = labels.view(batch_size * length).to(torch.int64)
        #
        # self.loss_hook(sentences, neg_samples, inputs)
        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_softmax_loss=loss)
        return loss, ret

class ReconstructionBERTSimSoftmaxLoss(nn.Module):
    name = 'reconstruct_bert_sim_softmax_loss'

    def __init__(self, vocab_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionBERTSimSoftmaxLoss, self).__init__()
        self.vocab_size = vocab_size
        self.cls = nn.Parameter(torch.FloatTensor(vocab_size, size))
        self._cuda = cuda
        self.reset_parameters()
    
    def reset_parameters(self):
        self.cls.data.normal_()
    
    def loss_hook(self, sentences, negative_samples, inputs):
        pass
    
    def forward(self, sentences, negative_samples, diora, info):
        # batch: batch_size x origin_len x emb_size
        # negative_samples: batch_size x origin_len x k_neg x origin_len x emb_size
        # => batch_size x 
        batch_size, origin_len, _ = sentences.shape
        k_neg = negative_samples.shape[2]

        size = diora.outside_h.shape[-1]

        # batch_size x origin_len x 1 x emb_size
        cell = diora.outside_h[:, :origin_len].view(batch_size, origin_len, -1)

        negative_samples = torch.diagonal(negative_samples, dim1=1, dim2=3)
        
        # print('negative_samples shape: ', negative_samples.shape)

        # batch_size x origin_len x k_neg x emb_size
        negative_samples = negative_samples.permute(0, 3, 1, 2)

        # print('negative_samples shape: ', negative_samples.shape)

        score_positive = torch.einsum('abc,abc->ab', sentences, cell).unsqueeze(-1)
        score_negative = torch.einsum('abkc,abc->abk', negative_samples, cell)

        # print('score_positive: ', score_positive.shape)
        # print('score_negative: ', score_negative.shape)

        scores = torch.cat([score_positive, score_negative], 2)
        
        lossfn = nn.CrossEntropyLoss()
        inputs = scores.view(batch_size * origin_len, k_neg + 1)
        
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruct_bert_sim_softmax_loss=loss)
        return loss, ret

class ReconstructionVizSoftmaxLoss(nn.Module):
    name = 'reconstruct_viz_softmax_loss'

    def __init__(self, embeddings, input_dim, size, margin=1, cuda=False):
        super(ReconstructionVizSoftmaxLoss, self).__init__()
        self.input_size = input_dim
        self.size = size

        self.embeddings = embeddings
        # for p in self.embeddings.parameters():
        #     p.requires_grad = False
        # get the embedding
        # self.nclusters = embeddings.model.nclusters
        # self.backbone_dim = embeddings.model.backbone_dim

        # self.cls = nn.Parameter(torch.FloatTensor(input_dim, size))
    
        # self.cls = nn.Parameter(torch.FloatTensor(self.nclusters, self.backbone_dim))
        self._cuda = cuda

        self.reset_parameters()
    
    def reset_parameters(self):
        # self.cls.data.normal_()
        pass
    
    def forward(self, batch, neg_samples, diora, info):
        # images = batch
        # batch_size, length, img_size, img_size = images.shape
        # input_size = self.input_size
        # size = diora.outside_h.shape[-1]

        ############################## Constractive Learning in one batch #########################
        # batch_size x length x emb_size
        # emb_pos: 16 x 6 x 400
        # emb_pos = self.embeddings(images, 'backbone')       
        # emb_pos = emb_pos.view(batch_size * length, -1)

        # # print('emb_pos shape: ', emb_pos.shape)
        # # batch_size x length x emb_size
        # # cell: 16 x 6 x 400
        # cell = diora.outside_h[:, :length].view(batch_size, length, -1)        
        # # cell = cell.view(batch_size * length, -1)
        # cell = cell.reshape(batch_size * length, -1)

        # features_pos = F.normalize(emb_pos, dim=1)
        # features_cell = F.normalize(cell, dim=1)

        # similarity_matrix = torch.matmul(features_pos, features_cell.T)

        # lossfn = nn.CrossEntropyLoss()
        # outputs = torch.arange(0, batch_size * length).cuda()
        
        # loss = lossfn(similarity_matrix, outputs)

        ########################### KL divergence ####################
        ## use the classifier
        # emb_pos = self.embeddings(images, 'default')
        # emb_pos = emb_pos.view(batch_size * length, -1)
        # emb_pos = F.softmax(emb_pos, dim=1)

        # print('emb_pos: ', emb_pos)

        # cell = diora.outside_h[:, :length].view(batch_size, length, -1)

        # cell = self.embeddings(cell, 'head')
        # cell = torch.matmul(cell, torch.t(self.cls))
        # cell = cell.reshape(batch_size * length, -1)
        # cell = F.log_softmax(cell, dim=1)
        # print('cell_pred: ', cell_pred)
        
        # kl divergency
        lossfn = nn.KLDivLoss()
        loss = lossfn(diora, batch)

        ######################### Classification #####################
        # emb_pos = self.embeddings(images, 'default')
        # emb_pos = emb_pos.view(batch_size * length, -1)
        # emb_pos = torch.argmax(emb_pos, dim=1)

        # cell = diora.outside_h[:, :length].view(batch_size, length, -1)
        # cell_pred = self.embeddings(cell, 'head')
        # cell_pred = cell_pred.view(batch_size * length, -1)

        # lossfn = nn.CrossEntropyLoss()
        # loss = lossfn(cell_pred, emb_pos)
        

        # emb_pos = self.embeddings(images, 'default')
        # emb_pos = emb_pos.view(batch_size * length, -1)
        
        # cell = diora.outside_h[:, :length].view(batch_size, length, -1)
        # cell_pred = self.embeddings(cell, 'head')
        # cell_pred = cell.pred_view(batch_size * length, -1)

        ret = dict(reconstruction_viz_softmax_loss=loss)

        # print('loss: ', loss)

        return loss, ret

def get_loss_funcs(options, batch_iterator=None, embedding_layer=None):
    bert_bool = (options.emb == 'bert')
    elmo_bool = (options.emb == 'elmo')
    viz_bool = (options.emb.startswith('res'))

    if bert_bool:
        if hasattr(options, 'vocab_size'):
            input_dim = options.vocab_size
    elif viz_bool:
        input_dim = embedding_layer.model.backbone_dim
    elif elmo_bool:
        input_dim = embedding_layer.weight.shape[1]
        
    
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    cuda = options.cuda

    loss_funcs = []

    # Reconstruction Loss
    if elmo_bool and options.reconstruct_mode == 'margin':
        reconstruction_loss_fn = ReconstructionLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)
    elif elmo_bool and options.reconstruct_mode == 'softmax':
        reconstruction_loss_fn = ReconstructionSoftmaxLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)
    elif options.k_neg <= 0 and bert_bool and options.reconstruct_mode == 'softmax':
        reconstruction_loss_fn = ReconstructionBERTSoftmaxLoss(vocab_size=input_dim, size=size, cuda=cuda)
    elif options.k_neg > 0 and bert_bool and options.reconstruct_mode == 'softmax' :
        reconstruction_loss_fn = ReconstructionBERTSimSoftmaxLoss(vocab_size=input_dim, size=size, cuda=cuda)
    elif viz_bool and options.reconstruct_mode == 'softmax':
        reconstruction_loss_fn = ReconstructionVizSoftmaxLoss(None, input_dim=input_dim, size=size, cuda=cuda)
    loss_funcs.append(reconstruction_loss_fn)

    return loss_funcs


class Embed(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super(Embed, self).__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, x):
        batch_size, length = x.shape
        e = self.embeddings(x.view(-1))
        t = torch.mm(e, self.mat.t()).view(batch_size, length, -1)
        return t

class BERTEmbed(nn.Module):
    def __init__(self, bert_model, input_size, size):
        super(BERTEmbed, self).__init__()
        self.input_size = input_size
        self.size = size
        self.bert_model = bert_model
        self.mat = nn.Linear(input_size, size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mat.weight)
        nn.init.normal_(self.mat.bias)
    
    def forward(self, batch):
        # print('kargs keys: ', batch.keys())
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        
        e = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        # print('e shape: ', e.shape)
        # e:  torch.Size([64, 22, 768])

        token_mask = batch['token_mask']
        # print('token_mask: ', token_mask.shape)
        # token_mask:  torch.Size([64, 20, 22])

        # mask
        e = torch.einsum('abc,adb->adc', e, token_mask)
        # print('e shape: ', e.shape)
        # print('e: ', e.shape)
        # print('e type: ', type(e))
        # print('self.mat: ', self.mat.shape)
        # t = torch.mm(e, self.mat.t())
        t = self.mat(e)
        # print('t shape: ', t.shape)
        return t

class VizEmbed(nn.Module):
    def __init__(self, resnet, input_size, size):
        super(VizEmbed, self).__init__()
        self.model = resnet
        self.input_size = input_size
        self.size = size
        self.mat = nn.Linear(input_size, size)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.mat.weight)
        nn.init.normal_(self.mat.bias)
    
    def forward(self, batch, forward_pass='backbone'):
        assert forward_pass in {'default', 'backbone', 'head', 'return_all'}

        dim_size = len(batch.shape)

        if dim_size == 3:
            batch_size, channel, img_size = batch.shape
            batch = batch.reshape(batch_size * channel, img_size)
        else:
            batch_size, channel, img_size, img_size = batch.shape
            batch = batch.view(batch_size * channel, 1, img_size, img_size)
        
        e = self.model(batch, forward_pass)

        if forward_pass in {'default', 'head'}:
            e = e[0]

        e = e.view(batch_size, channel, -1)

        return e

        # t = self.mat(e)

        # return t
    
    # def forward_without_mapping()
    

class Net(nn.Module):
    def __init__(self, embed, diora, loss_funcs=[]):
        super(Net, self).__init__()
        size = diora.size

        self.embed = embed
        self.diora = diora
        self.loss_func_names = [m.name for m in loss_funcs]

        for m in loss_funcs:
            setattr(self, m.name, m)

        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def compute_loss(self, batch, neg_samples, info):
        ret, loss = {}, []

        # Loss
        diora = self.diora.get_chart_wrapper()
        for func_name in self.loss_func_names:
            func = getattr(self, func_name)
            subloss, desc = func(batch, neg_samples, diora, info)
            loss.append(subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v

        loss = torch.cat(loss, 1)

        return ret, loss

    def forward(self, batch, neg_samples=None, compute_loss=True, info=None):
        # batch shape: batch_size x seq_len
        # Embed
        embed = self.embed(batch)
    
        # shape of embedding: batch_size x seq_len x emb_size
        # Run DIORA
        self.diora(embed)

        # Compute Loss
        if compute_loss:
            ret, loss = self.compute_loss(batch, neg_samples, info=info)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32,
                device=embed.device)

        # Results
        ret['total_loss'] = loss

        return ret

class BERT_Net(nn.Module):
    def __init__(self, bert, diora, loss_funcs=[]):
        super(BERT_Net, self).__init__()
        size = diora.size

        self.embed = bert
        self.diora = diora
        self.loss_func_names = [m.name for m in loss_funcs]

        for m in loss_funcs:
            setattr(self, m.name, m)
        
        # self.reset_parameters()
    
    def compute_loss(self, batch, neg_samples, info):
        ret, loss = {}, []

        # Loss
        diora = self.diora.get_chart_wrapper()
        for func_name in self.loss_func_names:
            func = getattr(self, func_name)
            subloss, desc = func(batch, neg_samples, diora, info)
            loss.append(subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v

        loss = torch.cat(loss, 1)

        return ret, loss

    def forward(self, batch, neg_samples=None, compute_loss=True, info=None):
        # batch shape: batch_size x seq_len
        # get stats
        batch_size = batch['input_ids'].shape[0]
        tk_len = batch['input_ids'].shape[1]

        # Embed
        embed = self.embed(batch)

        neg_emb = None

        if neg_samples is not None:
            # get origin_len, k_neg
            _, origin_len, k_neg, _ = neg_samples['input_ids'].shape
            # reshape
            neg_samples['input_ids'] = neg_samples['input_ids'].view(batch_size * origin_len * k_neg, tk_len)
            neg_samples['token_type_ids'] = neg_samples['token_type_ids'].view(batch_size * origin_len * k_neg, tk_len)
            neg_samples['attention_mask'] = neg_samples['attention_mask'].view(batch_size * origin_len * k_neg, tk_len)
            neg_samples['token_mask'] = neg_samples['token_mask'].view(batch_size * origin_len * k_neg, origin_len, tk_len)
            neg_emb = self.embed(neg_samples)
            neg_emb = neg_emb.view(batch_size, origin_len, k_neg, origin_len, -1)    
        # shape of embedding: batch_size x seq_len x emb_size
        # Run DIORA
        self.diora(embed)

        # Compute Loss
        if compute_loss:
            ret, loss = self.compute_loss(embed, neg_emb, info=info)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32,
                device=embed.device)

        # Results
        ret['total_loss'] = loss

        return ret

class Viz_Net(nn.Module):
    def __init__(self, viz_emb, diora, loss_funcs=[]):
        super(Viz_Net, self).__init__()
        size = diora.size
        self.embed = viz_emb
        self.diora = diora
        self.loss_func_names = [m.name for m in loss_funcs]

        for m in loss_funcs:
            setattr(self, m.name, m)

    def compute_loss(self, batch, neg_samples, info):
        ret, loss = {}, []

        batch_size, length, img_size, img_size = batch.shape

        diora = self.diora.get_chart_wrapper()

        cell = diora.outside_h[:, :length].view(batch_size, length, -1)
        cell = self.embed(cell, 'head')
        cell = cell.reshape(batch_size * length, -1)
        cell = F.log_softmax(cell, dim=1)

        emb_pos = self.embed(batch, 'default')
        emb_pos = emb_pos.view(batch_size * length, -1)
        emb_pos = F.softmax(emb_pos, dim=1)

        for func_name in self.loss_func_names:
            func = getattr(self, func_name)
            subloss, desc = func(emb_pos, None, cell, info)
            loss.append(subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v
        
        loss = torch.cat(loss, 1)

        return ret, loss
    
    def forward(self, batch, neg_samples=None, compute_loss=True, info=None):
        # embed = self.embed(batch, 'default')
        embed = self.embed(batch)
        # print('cls: ', self.embed.model.nclusters)
        # print('embed shape: ', embed.shape)

        self.diora(embed)

        if compute_loss:
            ret, loss = self.compute_loss(batch, neg_samples, info=info)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32, device=embed.device)
        
        ret['total_loss'] = loss

        return ret

    def get_class(self, batch):
        with torch.no_grad():
            # B x L x C
            output = self.embed(batch, 'default').detach().clone()
            # print('output: ', output)
            output_idx = torch.argmax(output, dim=-1)
        return output, output_idx


class Trainer(object):
    def __init__(self, options, net, k_neg=None, ngpus=None, cuda=None):
        super(Trainer, self).__init__()
        self.options = options
        self.net = net
        self.optimizer = None
        self.optimizer_cls = None
        self.optimizer_kwargs = None
        self.cuda = cuda
        self.ngpus = ngpus

        # update the categories
        self.bert_bool = (options.emb == 'bert')
        self.viz_bool = (options.emb == 'resnet')
        self.elmo_bool = (options.emb == 'elmo')

        self.parallel_model = None

        print("Trainer initialized with {} gpus.".format(ngpus))

    def freeze_diora(self):
        for p in self.net.diora.parameters():
            p.requires_grad = False

    def parameter_norm(self, requires_grad=True, diora=False):
        net = self.net.diora if diora else self.net
        total_norm = 0
        for p in net.parameters():
            if requires_grad and not p.requires_grad:
                continue
            total_norm += p.norm().item()
        return total_norm

    def init_optimizer(self, optimizer_cls, optimizer_kwargs):
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

    @staticmethod
    def get_single_net(net):
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            return net.module
        return net

    def save_model(self, model_file):
        state_dict = self.net.state_dict()

        todelete = []

        for k in state_dict.keys():
            if 'embeddings' in k:
                todelete.append(k)

        for k in todelete:
            del state_dict[k]

        torch.save({
            'state_dict': state_dict,
        }, model_file)

    @staticmethod
    def load_model(net, model_file):
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict_toload = save_dict['state_dict']
        state_dict_net = Trainer.get_single_net(net).state_dict()

        # Bug related to multi-gpu
        keys = list(state_dict_toload.keys())
        prefix = 'module.'
        for k in keys:
            if k.startswith(prefix):
                newk = k[len(prefix):]
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        # Remove extra keys.
        keys = list(state_dict_toload.keys())
        for k in keys:
            if k not in state_dict_net:
                print('deleting {}'.format(k))
                del state_dict_toload[k]

        # Hack to support embeddings.
        for k in state_dict_net.keys():
            if 'embeddings' in k:
                state_dict_toload[k] = state_dict_net[k]

        Trainer.get_single_net(net).load_state_dict(state_dict_toload)

    def run_net(self, batch_map, compute_loss=True, multigpu=False):
        batch = batch_map['sentences']
        neg_samples = batch_map.get('neg_samples', None)
        info = self.prepare_info(batch_map)
        out = self.net(batch, neg_samples=neg_samples, compute_loss=compute_loss, info=info)
        return out

    def run_bert_net(self, batch_map, compute_loss=True, multigpu=False):
        batch = batch_map['sentences']
        neg_samples = batch_map.get('neg_samples', None)

        info = self.prepare_info(batch_map)
        out = self.net(batch, neg_samples=neg_samples, compute_loss=compute_loss, info=info)
        return out

    def run_viz_net(self, batch_map, compute_loss=True, multigpu=False):
        batch = batch_map['samples']
        neg_samples = batch_map.get('neg_samples', None)

        info = self.prepare_info(batch_map)
        out = self.net(batch, neg_samples=neg_samples, compute_loss=compute_loss, info=info)
        return out

    def gradient_update(self, loss):
        # self.optimizer.zero_grad()
        # loss.backward()
        # params = [p for p in self.net.parameters() if p.requires_grad]
        # total_sum = [p.numel() for p in params]
        # # print(sum(total_sum))
        # torch.nn.utils.clip_grad_norm_(params, 5.0)
        # self.optimizer.step()
        pass

    def prepare_result(self, batch_map, model_output):
        result = {}
        result['batch_size'] = batch_map['batch_size']
        result['length'] = batch_map['length']
        for k, v in model_output.items():
            if 'loss' in k:
                result[k] = v.mean(dim=0).sum().item()
        return result

    def prepare_info(self, batch_map):
        return {}

    def step(self, *args, **kwargs):
        try:
            return self._step(*args, **kwargs)
        except Exception as err:
            batch_map = args[0]
            if self.ngpus > 1:
                print(traceback.format_exc())
                print('The step failed. Running multigpu cleanup.')
                os.system("ps -elf | grep [p]ython | grep adrozdov | grep " + self.experiment_name + " | tr -s ' ' | cut -f 4 -d ' ' | xargs -I {} kill -9 {}")
                sys.exit()
            else:
                raise err

    def _step(self, batch_map, train=True, compute_loss=True):
        if train:
            self.net.train()
        else:
            self.net.eval()
        multigpu = self.ngpus > 1 and train

        if self.elmo_bool:
            with torch.set_grad_enabled(train):
                model_output = self.run_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)
        elif self.viz_bool:
            with torch.set_grad_enabled(train):
                model_output = self.run_viz_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)
        elif self.bert_bool:
            with torch.set_grad_enabled(train):
                model_output = self.run_bert_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)
        
        # Calculate average loss for multi-gpu and sum for backprop.
        total_loss = model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)

        result = self.prepare_result(batch_map, model_output)

        return result

    def get_class(self, batch_map):
        if not self.viz_bool:
            raise Exception("No class for ELMO and BERT is not valid.")
        
        batch = batch_map['samples']
        return self.net.get_class(batch)


def build_net(options, embeddings=None, batch_iterator=None, random_seed=None):
    # bert_bool = isinstance(embeddings, transformers.models.bert.modeling_bert.BertModel)
    bert_bool = (options.emb == 'bert')
    viz_bool = (options.emb == 'resnet')
    elmo_bool = (options.emb == 'elmo')

    logger = get_logger()

    lr = options.lr
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    normalize = options.normalize

    for p in embeddings.parameters():
        if p.requires_grad:
            print('REQUIRES_GRAD')

    if bert_bool:
        input_dim = embeddings.encoder.layer[-1].output.dense.out_features
    elif elmo_bool:
        input_dim = embeddings.shape[1]
    elif viz_bool:
        input_dim = embeddings.backbone_dim
    else:
        raise Exception('The emb type {} is not valid.'.format(options.emb))
    
    cuda = options.cuda
    rank = options.local_rank
    ngpus = 1

    if cuda and options.multigpu:
        ngpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = options.master_addr
        os.environ['MASTER_PORT'] = options.master_port
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if bert_bool:
        embed = BERTEmbed(embeddings, input_size=input_dim, size=size)
    elif elmo_bool:
        embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
        embed = Embed(embedding_layer, input_size=input_dim, size=size)
    elif viz_bool:
        embed = VizEmbed(embeddings, input_size=input_dim, size=size)

    # DIORA
    if options.arch == 'treelstm':
        diora = DioraTreeLSTM(size, outside=True, normalize=normalize, compress=False)
    elif options.arch == 'mlp':
        diora = DioraMLP(size, outside=True, normalize=normalize, compress=False)
    elif options.arch == 'mlp-shared':
        diora = DioraMLPShared(size, outside=True, normalize=normalize, compress=False)

    # Loss
    # normally, it is ReconstructionLoss
    if bert_bool:
        loss_funcs = get_loss_funcs(options, batch_iterator)
    elif elmo_bool:
        loss_funcs = get_loss_funcs(options, batch_iterator, embedding_layer)
    elif viz_bool:
        loss_funcs = get_loss_funcs(options, batch_iterator, embed)

    # loss_funcs:  [ReconstructionSoftmaxLoss((embeddings): Embedding(97096, 1024))]

    # Net
    if elmo_bool:
        net = Net(embed, diora, loss_funcs=loss_funcs)
    elif bert_bool:
        net = BERT_Net(embed, diora, loss_funcs=loss_funcs)
    elif viz_bool:
        net = Viz_Net(embed, diora, loss_funcs=loss_funcs)

    # Load model.
    if options.load_model_path is not None:
        logger.info('Loading model: {}'.format(options.load_model_path))
        Trainer.load_model(net, options.load_model_path)

    # CUDA-support
    if cuda:
        if options.multigpu:
            torch.cuda.set_device(options.local_rank)
        net.cuda()
        diora.cuda()

    if cuda and options.multigpu:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[rank], output_device=rank)

    # Trainer
    trainer = Trainer(options, net, k_neg=k_neg, ngpus=ngpus, cuda=cuda)
    trainer.rank = rank
    trainer.experiment_name = options.experiment_name # for multigpu cleanup
    trainer.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    return trainer
