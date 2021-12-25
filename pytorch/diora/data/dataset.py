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

# MAX_SUBTKS = 50

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class ConsolidateDatasets(object):
    """
    A class for consolidating many datasets.
    """

    def __init__(self, datasets):
        # a list of dataset
        self.datasets = datasets

    def reindex(self, sentences, inverse_mapping):
        def fn(s):
            for idx in s:
                yield inverse_mapping[idx]
        def queue(lst):
            q = deque(lst)
            while len(q) > 0:
                yield q.popleft()
        return [list(fn(s)) for s in tqdm(queue(sentences), desc='reindex')]

    def remap_embeddings(self, datasets, inverse_mapping_lst, master_word2idx):
        size = datasets[0]['embeddings'].shape[1]
        embeddings = np.zeros((len(master_word2idx), size), dtype=np.float32)
        for dset, old2master in zip(datasets, inverse_mapping_lst):
            idx_from, idx_to = zip(*old2master.items())
            embeddings[np.asarray(idx_to)] = dset['embeddings'][np.asarray(idx_from)]
        return embeddings

    def consolidate_word2idx(self, word2idx_lst):
        master_word2idx = {}
        inverse_mapping_lst = []

        for w2i in word2idx_lst:
            old2master = {}
            for w, idx in w2i.items():
                if w not in master_word2idx:
                    master_word2idx[w] = len(master_word2idx)
                old2master[idx] = master_word2idx[w]
            inverse_mapping_lst.append(old2master)

        return master_word2idx, inverse_mapping_lst

    def run(self):
        # combine two data inforamtion such as [word2idx]
        word2idx_lst = [x['word2idx'] for x in self.datasets]
        master_word2idx, inverse_mapping_lst = self.consolidate_word2idx(word2idx_lst)
        embeddings = self.remap_embeddings(self.datasets, inverse_mapping_lst, master_word2idx)
        for dset, inverse_mapping in zip(self.datasets, inverse_mapping_lst):
            dset['sentences'] = self.reindex(dset['sentences'], inverse_mapping)
            dset['word2idx'] = master_word2idx
            dset['embeddings'] = embeddings


class ReaderManager(object):
    def __init__(self, reader):
        super(ReaderManager, self).__init__()
        self.reader = reader
        self.logger = get_logger()

    def run(self, options, text_path, embeddings_path):
        reader = self.reader
        logger = self.logger

        logger.info('Reading text: {}'.format(text_path))
        reader_result = reader.read(text_path)
        
        sentences = reader_result['sentences']
        # print('the first sent: ', sentences[0])
        # print('len: ', len(sentences))

        extra = reader_result['extra']
        # print('len extra: ', len(extra))
        metadata = reader_result.get('metadata', {})
        logger.info('len(sentences)={}'.format(len(sentences)))

        if options.word2idx is not None:
            # load the word2idx
            with open(options.word2idx, 'rb') as r:
                word2idx = pickle.load(r)

            word2idx = word2idx.word2idx
        else:
            word2idx = build_text_vocab(sentences)
        logger.info('len(vocab)={}'.format(len(word2idx)))

        embeddings = None
        if options.emb != 'bert':
            if 'embeddings' in metadata:
                logger.info('Using embeddings from metadata.')
                embeddings = metadata['embeddings']
                del metadata['embeddings']
            else:
                logger.info('Reading embeddings.')
                embeddings, word2idx = EmbeddingsReader().get_embeddings(
                    options, embeddings_path, word2idx)

        unk_index = word2idx.get(UNK_TOKEN, None)
        logger.info('Converting tokens to indexes (unk_index={}).'.format(unk_index))
        sentences = indexify(sentences, word2idx, unk_index)
        

        return {
            "sentences": sentences,
            "embeddings": embeddings,
            "word2idx": word2idx,
            "extra": extra,
            "metadata": metadata,
        }


class ReconstructDataset(object):

    def initialize(self, options, text_path=None, embeddings_path=None, filter_length=0, data_type=None):
        if data_type == 'nli':
            reader = NLIReader.build(lowercase=options.lowercase, filter_length=filter_length)
        elif data_type == 'conll_jsonl':
            reader = ConllReader(lowercase=options.lowercase, filter_length=filter_length)
        elif data_type == 'txt':
            reader = PlainTextReader(lowercase=options.lowercase, filter_length=filter_length, include_id=False)
        elif data_type == 'txt_id':
            reader = PlainTextReader(lowercase=options.lowercase, filter_length=filter_length, include_id=True)
        elif data_type == 'jsonl':
            reader = JSONLReader(lowercase=options.lowercase, filter_length=filter_length)
        elif data_type == 'synthetic':
            reader = SyntheticReader(nexamples=options.synthetic_nexamples,
                embedding_size=options.synthetic_embeddingsize,
                vocab_size=options.synthetic_vocabsize, seed=options.synthetic_seed,
                minlen=options.synthetic_minlen, maxlen=options.synthetic_maxlen,
                length=options.synthetic_length)
        elif data_type == 'partit':
            reader = PartItTextReader(lowercase=options.lowercase, filter_length=filter_length)
        elif data_type == 'partitwhole':
            reader = PartItWholeTextReader(lowercase=options.lowercase, filter_length=filter_length)

        manager = ReaderManager(reader)
        result = manager.run(options, text_path, embeddings_path)

        return result

def generate_inputs_subwords(sent, tokenizer, mask=True):
    # check the overall length first
    len_sent = len(sent)
    sent_join = ' '.join(sent)
    inputs = tokenizer(sent_join, return_tensors='pt')

    inputs['labels'] = inputs.input_ids.detach().clone()

    tks_len = len(inputs.input_ids[0].tolist())

    tokens = []

    tokens_mask = []
    
    start_pos = 1
    end_pos = start_pos

    for word in sent:
        word_tokens = tokenizer.tokenize(word)
        tk_len = len(word_tokens)
        end_pos = start_pos + tk_len
        tk_mask = [0. for _ in range(tks_len)]
        tk_mask[start_pos:end_pos] = [1./tk_len for _ in range(tk_len)]
        start_pos = end_pos

        tokens_mask.append(tk_mask)

    assert end_pos == tks_len-1, 'The last position should be the same as the sentence length.'

    
    # print('len(tokens_mask): ', len(tokens_mask))
    # padding the mask
    padding_num = tks_len - 2 - len(tokens_mask)
    # # print('padding_num: ', padding_num)
    assert padding_num >= 0, 'The length of masks should be fewer than the filter length.'

    for i in range(padding_num):
        tokens_mask.append([0. for _ in range (tks_len)])

    inputs['token_mask'] = torch.FloatTensor(tokens_mask)

    # get the labels
    inputs['labels'] = inputs.input_ids.detach().clone()

    if mask:
        # generate the mask
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
        selection = torch.flatten((mask_arr[0].nonzero())).tolist()

        inputs.input_ids[0, selection] = 103
        # length of input
        no_mask = set(range(len(inputs.input_ids))) - set(selection)
        inputs['labels'][0, list(no_mask)] = -100

    return inputs

def generate_inputs(sent, tokenizer):
    len_sent = len(sent)
    sent_join = ' '.join(sent)
    inputs = tokenizer(sent_join, return_tensors='pt')

    inputs['labels'] = inputs.input_ids.detach().clone()

    # generate the mask
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)

    selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
    inputs.input_ids[0, selection] = 103

    # length of inputs
    no_mask = set(range(len(inputs.input_ids))) - set(selection)
    inputs['labels'][0, list(no_mask)] = -100

    # x = sent
    # y = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())

    # if len(x) != len(y):
    #     print(sent_join)
    #     print(x)
    #     print(y)

    # assert len(sent) == len(inputs['labels']), 'All tokens should be inside'

    return inputs


class MLMDataset(object):
    def initialize(self, options, tokenizer, model, text_path, filter_length):
        if options.data_type == 'partitwhole':
            reader = PartItWholeTextReader(lowercase=options.lowercase, filter_length=filter_length)
        elif options.data_type == 'partit':
            reader = PartItTextReader(lowercase=options.lowercase, filter_length=filter_length)
        else:
            raise Exception("Bert only for partitwhole datatype.")

        # {'sentence': [list of string]}
        reader_result = reader.read(text_path)
        sentences = reader_result['sentences']
        extra = reader_result['extra']
        metadata = reader_result.get('metadata', {})

        # read the wordidx
        if options.word2idx is not None:
            # load the word2idx
            with open(options.word2idx, 'rb') as r:
                word2idx = pickle.load(r)

            word2idx = word2idx.word2idx

        # check all words in word2idx
        word_set = set([x for l in sentences for x in l])
        print('There is {} oov words'.format(len(word_set - set(word2idx.keys()))))

        # check the new words outside the tokenizer and put them in
        tokenizer_vocab = tokenizer.vocab
        new_vocab = []
        vocab_add_cnt = 0
        for word in list(word_set):
            if word not in tokenizer_vocab:
                print(word)
                new_vocab.append(word)
                vocab_add_cnt += 1
        
        print('The number of added vocab cnt is {}'.format(vocab_add_cnt))
        
        if len(new_vocab) != 0:
            tokenizer.add_tokens(new_vocab)
            # add new words to model
            model.resize_token_embeddings(len(tokenizer))

        # generate
        dataset = [
            generate_inputs(sen, tokenizer) for sen in sentences
        ]
        
        return {
            'sentences': dataset,
            'word2idx': word2idx,
            'extra': extra,
            'metadata': metadata,
        }


class BERTDataset(object):
    def initialize(self, options, tokenizer, model, text_path, filter_length):
        if options.data_type == 'partitwhole':
            reader = PartItWholeTextReader(lowercase=options.lowercase, filter_length=filter_length)
        elif options.data_type == 'partit':
            reader = PartItTextReader(lowercase=options.lowercase, filter_length=filter_length)
        else:
            raise Exception("Bert only for partitwhole datatype.")

        # {'sentence': [list of string]}
        reader_result = reader.read(text_path)
        sentences = reader_result['sentences']
        extra = reader_result['extra']
        metadata = reader_result.get('metadata', {})

        # read the wordidx
        if options.word2idx is not None:
            # load the word2idx
            with open(options.word2idx, 'rb') as r:
                word2idx = pickle.load(r)

            word2idx = word2idx.word2idx

        # check all words in word2idx
        word_set = set([x for l in sentences for x in l])
        print('There is {} oov words'.format(len(word_set - set(word2idx.keys()))))

        # generate
        dataset = [
            generate_inputs_subwords(sent, tokenizer, options.mask) for sent in sentences
        ]
        
        return {
            'sentences': dataset,
            'word2idx': word2idx,
            'extra': extra,
            'metadata': metadata,
        }


class CombineDataset(object):
    def initialize(self, options, tokenizer, model, text_path, filter_length):
        reconstruct_data = ReconstructDataset().initialize(options,
                                                    text_path=text_path,
                                                    filter_length=filter_length,
                                                    data_type=options.data_type)
        
        # mlm_data = MLMDataset().initialize(options,
                                            # tokenizer,
                                            # model,
                                            # text_path=text_path,
                                            # filter_length=filter_length)
        
        mlm_data = BERTDataset().initialize(options,
                                            tokenizer,
                                            model,
                                            text_path=text_path,
                                            filter_length=filter_length)


        word2idx = mlm_data['word2idx']
        # idx to words
        idx2word = {}

        options.vocab_size = len(word2idx)

        for k, v in word2idx.items():
            idx2word[v] = k

        # merge them
        reconstruct_sents = reconstruct_data['sentences']
        reconstruct_extra = reconstruct_data['extra']['example_ids']

        # print('len(reconstruct_sents): ', len(reconstruct_sents))
        # print('len(reconstruct_extra): ', len(reconstruct_extra))
        # print('reconstruct_sents: ', reconstruct_sents)
        # print('reconstruct_extra: ', reconstruct_extra)

        assert len(reconstruct_sents) == len(reconstruct_extra), 'The length of reconstruction should be the same.'

        mlm_sents = mlm_data['sentences']
        mlm_extra = mlm_data['extra']['example_ids']

        assert len(mlm_sents) == len(mlm_extra), 'The length of mlm should be the same.'

        # build mapping
        reconstruct_mapping = {}
        for (reconstruct_flag, reconstruct_sent) in zip(reconstruct_extra, reconstruct_sents):
            reconstruct_mapping[reconstruct_flag] = reconstruct_sent

        mlm_mapping = {}
        for (mlm_flag, mlm_sent) in zip(mlm_extra, mlm_sents):
            mlm_mapping[mlm_flag] = mlm_sent

        keys_reconstruct = reconstruct_mapping.keys()
        keys_mlm = mlm_mapping.keys()
        
        assert set(keys_reconstruct) == set(keys_mlm), 'The keys for reconstruct and mlm should be the same.'
        
        combine_list = []

        for k in list(keys_reconstruct):
            item_mlm = mlm_mapping[k]
            item_reconstruct = reconstruct_mapping[k]
            tks_len = item_mlm['input_ids'].shape[1]
            item_reconstruct = item_reconstruct + [-100 for _ in range(tks_len - 2 - len(item_reconstruct))]
            item_mlm['origin_input_ids'] = torch.Tensor(item_reconstruct).unsqueeze(0)

            # reverse back
            # sent_mlm = [idx2word[x] for x in item_mlm['origin_input_ids'][0].tolist()]
            # sent_recon = tokenizer.convert_ids_to_tokens(item_mlm['input_ids'][0].tolist())
            
            assert item_mlm['input_ids'].shape[1] == item_mlm['origin_input_ids'].shape[1] + 2, 'MLM and Inputs does not allign.'

            combine_list.append(item_mlm)

        return {
            'sentences': combine_list,
            'word2idx': word2idx,
            'extra': mlm_data['extra'],
            'metadata': None,
        }


# def make_batch_iterator_bert(options, dset, shuffle=True, include_partial=False, filter_length=0,
#                         batch_size=None, length_to_size=None):
#     sentences = dset['sentences']
#     word2idx = dset['word2idx']
#     extra = dset['extra']
#     metadata = dset['metadata']

#     cuda = options.cuda
#     multigpu = options.multigpu
#     ngpus = 1
    


#     # DIRTY HACK: Makes it easier to print examples later. Should really wrap this within the class.
#     batch_iterator.word2idx = word2idx

#     return batch_iterator


def make_batch_iterator(options, dset, shuffle=True, include_partial=False, filter_length=0,
                        batch_size=None, length_to_size=None):

    sentences = dset['sentences']
    word2idx = dset['word2idx']
    extra = dset['extra']
    metadata = dset['metadata']

    cuda = options.cuda
    multigpu = options.multigpu
    ngpus = 1
    if cuda and multigpu:
        ngpus = torch.cuda.device_count()

    vocab_size = len(word2idx)

    options.vocab_size = vocab_size

    if hasattr(options, 'bert_type') and options.bert_type.startswith('bert'):
        negative_sampler = None
        print('options.reconstruct_mode: ', options.reconstruct_mode)
        if options.reconstruct_mode in ('margin', 'softmax'):
            # get the sentences here
            origin_sents = [
                x['origin_input_ids'][0].tolist() for x in sentences
            ]
            freq_dist = calculate_freq_dist(origin_sents, vocab_size)
            negative_sampler = NegativeSampler(freq_dist=freq_dist, dist_power=options.freq_dist_power)
        
        batch_iterator = BERTBatchIterator(
            sentences, extra=extra, shuffle=shuffle, include_partial=include_partial,
            filter_length=filter_length, batch_size=batch_size, rank=options.local_rank,
            cuda=cuda, ngpus=ngpus, negative_sampler=negative_sampler,
            vocab=None, k_neg=options.k_neg,
            options_path=options.elmo_options_path,
            weights_path=options.elmo_weights_path,
            length_to_size=length_to_size,
            )
    
    else:
        negative_sampler = None
        if options.reconstruct_mode in ('margin', 'softmax'):
            freq_dist = calculate_freq_dist(sentences, vocab_size)
            # sample based word-freq
            negative_sampler = NegativeSampler(freq_dist=freq_dist, dist_power=options.freq_dist_power)
        
        vocab_lst = [w for w, _ in sorted(word2idx.items(), key=lambda x: x[1])]
        batch_iterator = BatchIterator(
            sentences, extra=extra, shuffle=shuffle, include_partial=include_partial,
            filter_length=filter_length, batch_size=batch_size, rank=options.local_rank,
            cuda=cuda, ngpus=ngpus, negative_sampler=negative_sampler,
            vocab=vocab_lst, k_neg=options.k_neg,
            options_path=options.elmo_options_path,
            weights_path=options.elmo_weights_path,
            length_to_size=length_to_size,
            )

    # DIRTY HACK: Makes it easier to print examples later. Should really wrap this within the class.
    batch_iterator.word2idx = word2idx

    return batch_iterator
