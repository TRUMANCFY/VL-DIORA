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

from diora.net.trainer import VizEmbed, Embed, Net, Viz_Net, get_loss_funcs

class CombineTrainer(object):
    def __init__(self, options, viz_net, txt_net, k_neg=None, ngpus=None, cuda=None):
        super(CombineTrainer, self).__init__()
        self.options = options
        self.viz_net = viz_net
        self.txt_net = txt_net
        self.optimizer = None
        self.optimizer_cls = None
        self.optimizer_kwargs = None
        self.cuda = cuda
        self.ngpus = ngpus

    def freeze_diora(self):
        for p in self.viz_net.diora.parameters():
            p.requires_grad = False
        
        for p in self.txt_net.diora.parameters():
            p.requires_grad = False
        
    def eval_viz_embed(self):
        self.viz_net.embed.eval()

    def eval_embed(self):
        self.eval_viz_embed()
    
    def eval_whole(self):
        self.viz_net.eval()

    def train(self):
        self.viz_net.train()
        self.txt_net.train()
    
    def init_optimizer(self, optimizer_cls, optimizer_kwargs):
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs
        params = [p for p in self.viz_net.parameters() if p.requires_grad] + \
                [p for p in self.txt_net.parameters() if p.requires_grad]
            
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
    
    @staticmethod
    def get_single_net(net):
        return net
    
    def save_model(self, model_file):
        viz_state_dict = self.viz_net.state_dict()
        txt_state_dict = self.txt_net.state_dict()

        # print('viz_state_dict: ')
        # print(viz_state_dict.keys())
        # print('txt_state_dict: ')
        # print(txt_state_dict.keys())

        # viz_todelete = []
        # for k in viz_state_dict.keys():
        #     if 'embeddings' in k:
        #         viz_todelete.append(k)
        
        # for k in viz_todelete:
        #     del viz_state_dict[k]
        
        # txt_todelete = []
        # for k in txt_state_dict.keys():
        #     if 'embeddings' in k:
        #         txt_todelete.append(k)
        
        # for k in txt_todelete:
        #     del txt_state_dict[k]

        # print('after remove keys: ')
        # print('viz_state_dict:')
        # print(viz_state_dict.keys())

        # print('txt_state_dict: ')
        # print(txt_state_dict.keys())

        torch.save({
            'txt_state_dict': txt_state_dict,
            'viz_state_dict': viz_state_dict,
        }, model_file)


    @staticmethod
    def load_model(viz_net, txt_net, model_file):
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        viz_state_dict_toload = save_dict['viz_state_dict']
        txt_state_dict_toload = save_dict['txt_state_dict']

        viz_state_dict_net = CombineTrainer.get_single_net(viz_net).state_dict()
        txt_state_dict_net = CombineTrainer.get_single_net(txt_net).state_dict()

        # viz_keys = list(viz_state_dict_toload.keys())
        # txt_keys = list(txt_state_dict_toload.keys())

        # for k in viz_keys:
        #     if k not in viz_state_dict_net:
        #         print('viz deleting {}'.format(k))
        #         del viz_state_dict_toload[k]
        
        # for k in viz_state_dict_net.keys():
        #     if 'embeddings' not in k:
        #         viz_state_dict_toload[k] = viz_state_dict_net[k]

        # for k in txt_keys:
        #     if k not in txt_state_dict_net:
        #         print('txt deleting {}'.format(k))
        #         del txt_state_dict_toload[k]
        
        # for k in txt_state_dict_net.keys():
        #     if 'embeddings' not in k:
        #         txt_state_dict_toload[k] = txt_state_dict_net[k]
            
        CombineTrainer.get_single_net(viz_net).load_state_dict(viz_state_dict_toload)
        CombineTrainer.get_single_net(txt_net).load_state_dict(txt_state_dict_toload)

    def run_net(self, batch_map, compute_loss=True, multigpu=False):
        sentences = batch_map['sentences']
        samples = batch_map['samples']
        neg_samples = batch_map.get('neg_samples', None)
        
        info = self.prepare_info(batch_map)

        if self.options.txt2img:
            txt_out = self.txt_net(sentences, neg_samples=neg_samples, compute_loss=compute_loss, info=info)
            txt_info = None
            if self.options.mixture:
                txt_inside_info = self.txt_net.diora.inside_h
                txt_outside_info = None
                if self.options.outside_attn:
                    txt_outside_info = self.txt_net.diora.outside_h

                    assert '|'.join([str(x) for x in list(txt_outside_info.shape)]) == '|'.join([str(x) for x in list(txt_outside_info.shape)])
            # print('txt_info: ', txt_info.shape)
            viz_out = self.viz_net(samples, neg_samples=None, compute_loss=compute_loss, info=info, inside_info=txt_inside_info, outside_info=txt_outside_info)
        
        else:
            viz_out = self.viz_net(samples, neg_samples=None, compute_loss=compute_loss, info=info)
            viz_info = None
            if self.options.mixture:
                viz_inside_info = self.viz_net.diora.inside_h
                viz_outside_info = None
                if self.options.outside_attn:
                    viz_outside_info = self.viz_net.diora.outside_h

                    assert '|'.join([str(x) for x in list(viz_inside_info.shape)]) == '|'.join([str(x) for x in list(viz_outside_info.shape)]), 'The shape of viz between inside and outside is the same.'

            txt_out = self.txt_net(sentences, neg_samples=neg_samples, compute_loss=compute_loss, info=info, inside_info=viz_inside_info, outside_info=viz_outside_info)
        
        return txt_out, viz_out

    def run_txt_net(self, batch_map, compute_loss=True, multigpu=False):
        sentences = batch_map['sentences']
        neg_samples = batch_map.get('neg_samples', None)
        info = self.prepare_info(batch_map)

        txt_out = self.txt_net(sentences, neg_samples=neg_samples, compute_loss=compute_loss, info=info)
        return txt_out

    def run_viz_net(self, batch_map, compute_loss=True, multigpu=False):
        samples = batch_map['samples']
        info = self.prepare_info(batch_map)

        viz_out = self.viz_net(samples, neg_samples=None, compute_loss=compute_loss, info=info)
        return viz_out

    def prepare_result(self, batch_map, model_output):
        result = {}
        result['batch_size'] = batch_map['batch_size']
        result['txt_len'] = batch_map['txt_len']
        result['viz_len'] = batch_map['viz_len']
        for k, v in model_output.items():
            if 'loss' in k:
                result[k] = v.mean(dim=0).sum().item()
        
        return result

    def prepare_info(self, batch_map):
        return {}

    def gradient_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        params = [p for p in self.txt_net.parameters() if p.requires_grad] + \
                 [p for p in self.viz_net.parameters() if p.requires_grad]
                
        total_sum = [p.numel() for p in params]
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        self.optimizer.step()

    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)
    
    def _step(self, batch_map, train=True, compute_loss=True):
        if train:
            self.viz_net.train()
            self.txt_net.train()
            
            self.viz_net.embed.eval()
        else:
            self.viz_net.eval()
            self.txt_net.eval()

        multigpu = False

        with torch.set_grad_enabled(train):
            txt_model_output, viz_model_output = self.run_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)
        
        total_loss = txt_model_output['total_loss'].mean(dim=0).sum() + viz_model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)
        
        txt_result = self.prepare_result(batch_map, txt_model_output)
        viz_result = self.prepare_result(batch_map, viz_model_output)

        return (txt_result, viz_result)

    def step_txt(self, batch_map, train=True, compute_loss=True):
        if train:
            self.viz_net.train()
            self.txt_net.train()
            self.viz_net.embed.eval()
        else:
            self.viz_net.eval()
            self.txt_net.eval()

        multigpu = False

        with torch.set_grad_enabled(train):
            txt_model_output = self.run_txt_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)
        
        total_loss = txt_model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)
        
        txt_result = self.prepare_result(batch_map, txt_model_output)
        
        return txt_result

    def step_viz(self, batch_map, train=True, compute_loss=True):
        if train:
            self.viz_net.train()
            self.txt_net.train()
            self.viz_net.embed.eval()
        else:
            self.viz_net.eval()
            self.txt_net.eval()
        
        multigpu = False

        with torch.set_grad_enabled(train):
            viz_model_output = self.run_viz_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)
        
        total_loss = viz_model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)

        viz_result = self.prepare_result(batch_map, viz_model_output)
        
        return viz_result
        
    def eval(self):
        self.viz_net.eval()
        self.txt_net.eval()
    
    def get_class(self, batch_map):
        samples = batch_map['samples']
        return self.viz_net.get_class(samples) 

def build_net(options, viz_embedding, txt_embedding, batch_iterator=None, random_seed=None):
    logger = get_logger()

    lr = options.lr
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    normalize = options.normalize
    level_attn = options.level_attn
    outside_attn = options.outside_attn

    # get the text and vision embedding sizes
    txt_input_dim = txt_embedding.shape[1]
    viz_input_dim = viz_embedding.backbone_dim

    assert viz_input_dim == size, 'The vision embedding size is not equal to diora size.'

    cuda = options.cuda
    rank = options.local_rank
    ngpus = 1

    # get vision embedding
    viz_embed = VizEmbed(viz_embedding, input_size=viz_input_dim, size=size)
    txt_embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(txt_embedding), freeze=True)
    txt_embedding = Embed(txt_embedding_layer, input_size=txt_input_dim, size=size)

    if not options.diora_shared:
        txt_diora = DioraMLPShared(size, outside=True, normalize=normalize, compress=False, level_attn=level_attn)
        viz_diora = DioraMLPShared(size, outside=True, normalize=normalize, compress=False, level_attn=level_attn)

        txt_loss_funcs = get_loss_funcs(options, batch_iterator, txt_embedding_layer, 'elmo')
        viz_loss_funcs = get_loss_funcs(options, batch_iterator, viz_embed, 'resnet')

        txt_net = Net(txt_embedding, txt_diora, loss_funcs=txt_loss_funcs)
        viz_net = Viz_Net(viz_embed, viz_diora, loss_funcs=viz_loss_funcs)
    else:
        diora = DioraMLPShared(size, outside=True, normalize=normalize, compress=False, level_attn=level_attn)

        txt_loss_funcs = get_loss_funcs(options, batch_iterator, txt_embedding_layer, 'elmo')
        viz_loss_funcs = get_loss_funcs(options, batch_iterator, viz_embed, 'resnet')

        txt_net = Net(txt_embedding, diora, loss_funcs=txt_loss_funcs)
        viz_net = Viz_Net(viz_embed, diora, loss_funcs=viz_loss_funcs)
    
    if options.load_model_path is not None:
        logger.info('Loading model: {}'.format(options.load_model_path))
        CombineTrainer.load_model(viz_net, txt_net, options.load_model_path)
    
    if cuda:
        if not options.diora_shared:
            txt_diora.cuda()
            viz_diora.cuda()
        else:
            diora.cuda()
        txt_net.cuda()
        viz_net.cuda()

    trainer = CombineTrainer(options, viz_net, txt_net, k_neg=k_neg, ngpus=ngpus, cuda=cuda)
    trainer.rank = rank
    trainer.experiment_name = options.experiment_name
    trainer.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    return trainer
