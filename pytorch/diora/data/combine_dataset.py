from collections import deque

import torch
import pickle
import numpy as np

from tqdm import tqdm

from diora.data.reading import NLIReader, PlainTextReader, ConllReader, JSONLReader, PartItTextReader, PartItWholeTextReader
from diora.data.batch_iterator import BatchIterator
from diora.data.bert_batch_iterator import BERTBatchIterator
from diora.data.embeddings import EmbeddingsReader, UNK_TOKEN
from diora.data.preprocessing import indexify, build_text_vocab
from diora.data.preprocessing import synthesize_training_data
from diora.logging.configuration import get_logger
from diora.blocks.negative_sampler import NegativeSampler, calculate_freq_dist
from diora.blocks.negative_sampler import choose_negative_samples

import json
from torchvision import transforms
import PIL.Image as Image
from PIL import ImageOps

from diora.data.dataloader import FixedLengthBatchSampler, CombineDataset

from diora.data.dataloader import VizDataset

import os

def get_config(config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    return config
        

def make_combine_batch_iterator(options, dset, shuffle=True, include_partial=False, filter_length=0,
                            batch_size=None, length_to_size=None):

    print('dset: ', dset.keys())
    samples = dset['samples']
    sentences = dset['sentences']
    extra = dset['extra']
    metadata = dset['metadata']
    word2idx = dset['word2idx']
    
    tokenizer = None
    if 'tokenizer' in dset:
        tokenizer = dset['tokenizer']

    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(idx2word)

    cuda = options.cuda
    multigpu = options.multigpu
    ngpus = 1

    if options.reconstruct_mode in ('margin', 'softmax'):
        freq_dist = calculate_freq_dist(sentences, vocab_size)
        negative_sampler = NegativeSampler(freq_dist=freq_dist, dist_power=options.freq_dist_power)
        batch_iterator = CombineBatchIterator(
            samples=samples, sentences=sentences, extra=extra, shuffle=shuffle, include_partial=include_partial,
            filter_length=filter_length, batch_size=batch_size, rank=options.local_rank,
            cuda=cuda, ngpus=ngpus, negative_sampler=negative_sampler,
            vocab=None, k_neg=options.k_neg)

    return batch_iterator


def get_default_config():

    default_config = dict(
        batch_size=16,
        forever=False,
        drop_last=False,
        sort_by_length=True,
        shuffle=True,
        random_seed=None,
        filter_length=None,
        workers=0,
        pin_memory=False,
        include_partial=False,
        cuda=False,
        ngpus=1,
        k_neg=3,
        negative_sampler=None,
        options_path=None,
        weights_path=None,
        vocab=None,
        length_to_size=None,
        rank=None,
        emb_type='elmo',
    )

    return default_config

class CombineBatchIterator(object):
    
    def __init__(self, samples, sentences, extra, **kwargs):
        self.samples = samples
        self.sentences = sentences
        self.extra = extra
        self.config = config = get_config(get_default_config(), **kwargs)
        self.loader = None
    
    def get_dataset_size(self):
        return len(self.samples)
    
    def choose_negative_samples(self, negative_sampler, k_neg):
        return choose_negative_samples(negative_sampler, k_neg)
    
    def get_iterator(self, **kwargs):
        config = get_config(self.config.copy(), **kwargs)

        random_seed = config.get('random_seed')
        batch_size = config.get('batch_size')
        filter_length = config.get('filter_length')
        pin_memory = config.get('pin_memory')
        include_partial = config.get('include_partial')
        cuda = config.get('cuda')
        ngpus = config.get('ngpus')
        rank = config.get('rank')
        negative_sampler = config.get('negative_sampler', None)
        workers = config.get('workers')
        length_to_size = config.get('length_to_size', None)
        emb_type = config.get('emb_type', 'elmo')
        k_neg = config.get('k_neg')

        def collate_fn(batch):
            index, sents, samples = zip(*batch)
            sents = torch.from_numpy(np.array(sents)).long()
            
            images = [sample['images'] for sample in samples]
            targets = [sample['targets'] for sample in samples]
            
            batch_map = {}
            batch_map['index'] = index
            batch_map['samples'] = torch.stack(images, dim=0)
            batch_map['targets'] = torch.stack(targets, dim=0)

            batch_map['sents'] = sents

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]
            
            return batch_map
        
        if self.loader is None:
            rng = np.random.RandomState(seed=random_seed)
            dataset = VizDataset(self.samples)
            dataset = CombineDataset(self.sentences, self.samples)
            sampler = FixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
                maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size,
                emb_type='both')
            
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=workers, pin_memory=pin_memory, batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        
        def myiterator():
            for batch in self.loader:
                index = batch['index']
                sentences = batch['sents']
                samples = batch['samples']
                targets = batch['targets']

                batch_size, txt_len = sentences.shape

                batch_size, viz_len, _, _ = samples.shape

                neg_samples = None
                if negative_sampler is not None:
                    neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

                if cuda:
                    samples = samples.cuda()
                    sentences = sentences.cuda()
                    if neg_samples is not None:
                        neg_samples = neg_samples.cuda()

                batch_map = {}
                batch_map['sentences'] = sentences
                batch_map['samples'] = samples
                batch_map['targets'] = targets
                batch_map['neg_samples'] = neg_samples
                batch_map['batch_size'] = batch_size
                batch_map['txt_len'] = txt_len
                batch_map['viz_len'] = viz_len

                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()

        
        
        

            

