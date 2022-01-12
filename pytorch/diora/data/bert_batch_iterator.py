from diora.data.dataloader import FixedLengthBatchSampler, SimpleDataset
from diora.blocks.negative_sampler import choose_negative_samples

import torch
import numpy as np


def generate_inputs_worigin(sent, tokenizer, sent_tks, mask=True):
    # check the overall length
    word_len = len(sent_tks)

    assume_word_len = word_len + 14

    len_sent = len(sent)
    sent_join = ' '.join(sent)
    inputs = tokenizer(sent_join, return_tensors='pt')

    inputs['labels'] = inputs.input_ids.detach().clone()

    tks_len = len(inputs.input_ids[0].tolist())

    assert tks_len <=  assume_word_len, 'The assumed word length {} is shorter than Bert tks_len {}, sent: {}'.format(assume_word_len, tks_len, sent_join)

    addition_tks_len = assume_word_len - tks_len

    tokens = []
    tokens_mask = []

    start_pos = 1
    end_pos = start_pos

    for word in sent:
        word_tokens = tokenizer.tokenize(word)
        tk_len = len(word_tokens)
        end_pos = start_pos + tk_len
        tk_mask = [0. for _ in range(assume_word_len)]
        tk_mask[start_pos:end_pos] = [1./tk_len for _ in range(tk_len)]
        start_pos = end_pos

        tokens_mask.append(tk_mask)

    assert end_pos == tks_len-1, 'The last position should be the same as the sentence length.'

    inputs['token_mask'] = torch.FloatTensor(tokens_mask)

    if mask:
        # generate the mask
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
        selection = torch.flatten((mask_arr[0].nonzero())).tolist()
        inputs.input_ids[0, selection] = 103

        no_mask = set(range(len(inputs.input_ids)))
        inputs['labels'][0, list(no_mask)] = -100
    
    # padding to
    inputs['input_ids'] = torch.LongTensor(inputs['input_ids'][0].tolist() + [0 for _ in range(addition_tks_len)]).unsqueeze(0)
    inputs['token_type_ids'] = torch.LongTensor(inputs['token_type_ids'][0].tolist() + [0 for _ in range(addition_tks_len)]).unsqueeze(0)
    inputs['attention_mask'] = torch.LongTensor(inputs['attention_mask'][0].tolist() + [0 for _ in range(addition_tks_len)]).unsqueeze(0)
    inputs['labels'] = torch.LongTensor(inputs['labels'][0].tolist() + [-100 for _ in range(addition_tks_len)]).unsqueeze(0)

    # add seq
    inputs['origin_input_ids'] = torch.LongTensor(sent_tks).unsqueeze(0)

    inputs['sents'] = sent

    return inputs


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
        emb_type='elmo',
        word2idx=None,
        idx2word=None,
        tokenizer=None,
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
        emb_type = config.get('emb_type', 'elmo')
        
        word2idx = config.get('word2idx', None)
        idx2word = config.get('idx2word', None)
        tokenizer = config.get('tokenizer', None)

        print('k_neg: ', k_neg)

        def collate_fn(batch):
            # list of indexes and list of 
            index, sents = zip(*batch)
            
            ensembles = {}

            keys = list(sents[0].keys())

            batchsize = len(sents)

            for key in keys:
                if key == 'sents':
                    continue
                ensembles[key] = []
                for i in range(batchsize):
                    ensembles[key].append(sents[i][key])

                # stack list of tensor
                ensembles[key] = torch.stack(ensembles[key], 0).squeeze(1)
            
            ensembles['sents'] = [
                sent['sents'] for sent in sents
            ]

            neg_samples = None
            neg_samples_list = None
            if negative_sampler is not None:
                neg_samples = self.choose_negative_samples(negative_sampler, k_neg)
                neg_samples_list = neg_samples.tolist()
            

            if k_neg > 0:
                origin_len = ensembles['origin_input_ids'].shape[1]
                origin_input_ids = ensembles['origin_input_ids']
                input_tk_len = ensembles['input_ids'].shape[1]

                # origin_len x k_neg
                neg_sample_frame = []
                for bz_idx in range(batchsize):
                    origin_sent_ids = origin_input_ids[bz_idx].detach().clone().tolist()
                    pos_level_tmp = []
                    for idx in range(origin_len):
                        tk_level_tmp = []
                        for sub_tk in neg_samples_list:
                            sub_sent_ids = [x for x in origin_sent_ids]
                            sub_sent_ids[idx] = sub_tk
                            # convert it to sent
                            sub_sent = [idx2word[x] for x in sub_sent_ids]

                            sub_sample = generate_inputs_worigin(sub_sent, tokenizer, sub_sent_ids, False)
                            assert sub_sample['input_ids'].shape[1] == input_tk_len, 'sub_sample: {}/ input_tk_len: {}'.format(sub_sample['input_ids'].shape[1], input_tk_len)
                            tk_level_tmp.append(sub_sample)
                        pos_level_tmp.append(tk_level_tmp)
                    neg_sample_frame.append(pos_level_tmp)
                
                neg_ensembles = {}
                # concat togather
                neg_ensembles['input_ids'] = torch.zeros(batchsize, origin_len, k_neg, input_tk_len)
                neg_ensembles['token_type_ids'] = torch.zeros(batchsize, origin_len, k_neg, input_tk_len)
                neg_ensembles['attention_mask'] = torch.zeros(batchsize, origin_len, k_neg, input_tk_len)
                neg_ensembles['token_mask'] = torch.zeros(batchsize, origin_len, k_neg, origin_len, input_tk_len)

                for batch_idx in range(batchsize):
                    for idx in range(origin_len):
                        for sub_tk_idx in range(k_neg):
                            neg_ensembles['input_ids'][batch_idx, idx, sub_tk_idx] = neg_sample_frame[batch_idx][idx][sub_tk_idx]['input_ids'].squeeze(0)
                            neg_ensembles['token_type_ids'][batch_idx, idx, sub_tk_idx] = neg_sample_frame[batch_idx][idx][sub_tk_idx]['token_type_ids'].squeeze(0)
                            neg_ensembles['attention_mask'][batch_idx, idx, sub_tk_idx] = neg_sample_frame[batch_idx][idx][sub_tk_idx]['attention_mask'].squeeze(0)
                            neg_ensembles['token_mask'][batch_idx, idx, sub_tk_idx] = neg_sample_frame[batch_idx][idx][sub_tk_idx]['token_mask'].squeeze(0)
            
            # print('ensembles negative_token_mask: ', ensembles['negative_token_mask'].shape)
            # print('ensembles negative_token_type_ids: ', ensembles['negative_token_type_ids'].shape)
            # print('ensembles negative_attention_mask: ', ensembles['negative_attention_mask'].shape)
            # print('ensembles negative_token_mask: ', ensembles['negative_token_mask'].shape)

            ensembles['neg_ensembles'] = neg_ensembles

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
            sampler = FixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
                maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size,
                emb_type=emb_type)
            
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=workers, pin_memory=pin_memory,batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():
            # print('Inside the iterator')

            for batch in self.loader:
                # print('batch: ', batch)
                index = batch['index']
                sentences = batch['sents']

                batch_size, length = sentences['input_ids'].shape

                # neg_samples = None
                # if negative_sampler is not None:
                #     neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

                ## We need to regenerate the negative
                # print('negative_samples: ', neg_samples)

                neg_samples = None

                if cuda:
                    sentences['input_ids'] = sentences['input_ids'].cuda()
                    sentences['token_type_ids'] = sentences['token_type_ids'].cuda()
                    sentences['attention_mask'] = sentences['attention_mask'].cuda()
                    sentences['labels'] = sentences['labels'].cuda()
                    sentences['token_mask'] = sentences['token_mask'].cuda()
                    sentences['origin_input_ids'] = sentences['origin_input_ids'].cuda()
                    
                    if k_neg > 0:
                        neg_samples = {}
                        sentences['neg_ensembles']['input_ids'] = sentences['neg_ensembles']['input_ids'].cuda().long()
                        sentences['neg_ensembles']['token_type_ids'] = sentences['neg_ensembles']['token_type_ids'].cuda().long()
                        sentences['neg_ensembles']['attention_mask'] = sentences['neg_ensembles']['attention_mask'].cuda().long()
                        sentences['neg_ensembles']['token_mask'] = sentences['neg_ensembles']['token_mask'].cuda()

                batch_map = {}
                batch_map['sentences'] = sentences
                batch_map['neg_samples'] = sentences['neg_ensembles']
                batch_map['batch_size'] = batch_size
                batch_map['length'] = length


                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()
