from diora.data.dataloader import FixedLengthBatchSampler, SimpleDataset, BERTFixedLengthBatchSampler
from diora.blocks.negative_sampler import choose_negative_samples

from allennlp.modules.elmo import Elmo, batch_to_ids

import torch
import numpy as np


def get_config(config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    return config


def get_default_config():

    default_config = dict(
        batch_size=16,
        forever=False,
        drop_last=False,
        sort_by_length=True,
        shuffle=True,
        random_seed=None,
        filter_length=None,
        workers=10,
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
    )

    return default_config


class BERTBatchIterator(object):

    def __init__(self, sentences, extra={}, **kwargs):
        self.sentences = sentences
        self.config = config = get_config(get_default_config(), **kwargs)
        self.extra = extra
        self.loader = None

    def get_dataset_size(self):
        return len(self.sentences)

    def get_dataset_minlen(self):
        return min(map(len, self.sentences))

    def get_dataset_maxlen(self):
        return max(map(len, self.sentences))

    def get_dataset_stats(self):
        return 'size={} minlen={} maxlen={}'.format(
            self.get_dataset_size(), self.get_dataset_minlen(), self.get_dataset_maxlen()
        )

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
        k_neg = config.get('k_neg')
        negative_sampler = config.get('negative_sampler', None)
        workers = config.get('workers')
        length_to_size = config.get('length_to_size', None)

        def collate_fn(batch):
            # list of indexes and list of 
            index, sents = zip(*batch)
            
            ensembles = {}

            keys = list(sents[0].keys())

            batchsize = len(sents)

            for key in keys:
                ensembles[key] = []
                for i in range(batchsize):
                    ensembles[key].append(sents[i][key])
            
                # stack list of tensor
                ensembles[key] = torch.stack(ensembles[key], 0).squeeze(1)

            batch_map = {}
            batch_map['index'] = index
            batch_map['sents'] = ensembles

            # print('batch_map: ', sents[0])
            # print('batch_map lens: ', [len(x) for x in sents])

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]

            return batch_map

        if self.loader is None:
            rng = np.random.RandomState(seed=random_seed)
            # return index, sentence
            dataset = SimpleDataset(self.sentences)
            # only get the index
            sampler = BERTFixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
                maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size)
            
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=workers, pin_memory=pin_memory,batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():
            # print('Inside the iterator')

            for batch in self.loader:
                # print('batch: ', batch)
                index = batch['index']
                sentences = batch['sents']

                batch_size, length = sentences['input_ids'].shape

                neg_samples = None
                if negative_sampler is not None:
                    neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

                if cuda:
                    sentences['input_ids'] = sentences['input_ids'].cuda()
                    sentences['token_type_ids'] = sentences['token_type_ids'].cuda()
                    sentences['attention_mask'] = sentences['attention_mask'].cuda()
                    sentences['labels'] = sentences['labels'].cuda()
                    sentences['token_mask'] = sentences['token_mask'].cuda()
                    sentences['origin_input_ids'] = sentences['origin_input_ids'].cuda()

                batch_map = {}
                batch_map['sentences'] = sentences
                batch_map['neg_samples'] = neg_samples
                batch_map['batch_size'] = batch_size
                batch_map['length'] = length



                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()
