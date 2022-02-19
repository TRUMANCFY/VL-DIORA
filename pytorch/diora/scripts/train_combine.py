import argparse
import datetime
import math
import os
import random
import sys
import uuid

import torch
import torch.nn as nn

from diora.data.dataset import ConsolidateDatasets, ReconstructDataset, make_batch_iterator, Vocabulary, CombineDataset, CombineBertDataset
from diora.data.viz_dataset import VisionDataset
from diora.data.combine_dataset import make_combine_batch_iterator

from diora.utils.path import package_path
from diora.logging.configuration import configure_experiment, get_logger
from diora.utils.flags import stringify_flags, init_with_flags_file, save_flags
from diora.utils.checkpoint import save_experiment

from diora.net.experiment_logger import ExperimentLogger

from transformers import AutoTokenizer, AutoModel

from diora.net.vision_models import get_model

import pickle

import numpy as np

data_types_choices = ('nli', 'conll_jsonl', 'txt', 'txt_id', 'synthetic', 'jsonl', 'partit', "partitwhole", "viz")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])

def build_net(options, viz_embeddings, txt_embeddings, batch_iterator=None):
    from diora.net.trainer_combine import build_net

    trainer = build_net(options, viz_embeddings, txt_embeddings, batch_iterator, random_seed=options.seed)

    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.viz_net) + count_params(trainer.txt_net)))

    return trainer

def combine_viz_text_dataset(viz_dataset, text_dataset):
    text_dict = {}
    viz_dict = {}

    for idx in range(len(text_dataset['extra']['example_ids'])):
        example_id = text_dataset['extra']['example_ids'][idx]
        sent = text_dataset['sentences'][idx]
        text_dict[example_id] = sent

    for idx in range(len(viz_dataset['extra']['example_ids'])):
        example_id = viz_dataset['extra']['example_ids'][idx]
        img = viz_dataset['samples'][idx]
        viz_dict[example_id] = img

    res = {
        'sentences': list(text_dict.values()),
        'samples': list(viz_dict.values()),
        'embeddings':  text_dataset['embeddings'],
        'word2idx': text_dataset['word2idx'],
        'metadata': text_dataset['metadata'],
        'extra': {'example_ids': list(text_dict.keys())},
    }
    
    return res


def generate_seeds(n, seed=11):
    random.seed(seed)
    seeds = [random.randint(0, 2**16) for _ in range(n)]
    return seeds

def sum_params(params):
    # input is a list of params
    total_sum = 0.0
    for p in params:
        total_sum += np.sum(p.cpu().numpy())
    return total_sum

def run_train(options, train_iterator, trainer, validation_iterator):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    logger.info('Running train.')

    # seeds = generate_seeds(options.max_epoch, options.seed)
    seed = options.seed

    step = 0

    patience = 3
    best_loss = float('inf')  
    no_improve_cnt = 0 

    loss_records = []

    ids_pred_class_mapping = {}

    ids_image_parsing_mapping = {}
    
    # switch the vision 
    if options.freeze_model:
        print('The vision model has been fronzen.')
        trainer.eval_embed()

    for epoch in range(options.max_epoch):
        # --- Train--- #

        # seed = seeds[epoch]

        logger.info('epoch={} seed={}'.format(epoch, seed))

        def myiterator():
            it = train_iterator.get_iterator(random_seed=seed)
            count = 0

            for batch_map in it:
                # TODO: Skip short examples (optionally).
                if batch_map['txt_len'] <= 2:
                    continue
                
                if batch_map['viz_len'] <= 2:
                    continue

                yield count, batch_map
                count += 1

        batch_cnt = 0
        txt_cur_loss = 0.
        viz_cur_loss = 0.

        for batch_idx, batch_map in myiterator():

            # print('batch_map: ', batch_map.keys())
            
            txt_result, viz_result = trainer.step(batch_map)

            image_tmp = batch_map['samples'].detach().clone().cpu()

            batch_size = batch_map['batch_size']
                
            with torch.no_grad():
                pred_vectors, pred_classes = trainer.get_class(batch_map)


            batch_cnt += txt_result['batch_size']
            txt_cur_loss += txt_result['total_loss'] * txt_result['batch_size']
            viz_cur_loss += viz_result['total_loss'] * viz_result['batch_size']
            
            experiment_logger.record(txt_result)
            experiment_logger.record(viz_result)

            if step % options.log_every_batch == 0:
                experiment_logger.log_batch(epoch, step, batch_idx, batch_size=options.batch_size)

            # -- Periodic Checkpoints -- #

            if not options.multigpu or options.local_rank == 0:
                if step % options.save_latest == 0 and step >= options.save_after:
                    logger.info('Saving model (periodic).')
                    trainer.save_model(os.path.join(options.experiment_path, 'model_periodic.pt'))
                    save_experiment(os.path.join(options.experiment_path, 'experiment_periodic.json'), step)

                if step % options.save_distinct == 0 and step >= options.save_after:
                    logger.info('Saving model (distinct).')
                    trainer.save_model(os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)))
                    save_experiment(os.path.join(options.experiment_path, 'experiment.step_{}.json'.format(step)), step)

            del txt_result
            del viz_result

            step += 1

        experiment_logger.log_epoch(epoch, step)
        txt_avg_loss = txt_cur_loss / batch_cnt
        viz_avg_loss = viz_cur_loss / batch_cnt
        logger.info('The txt avg_loss is: {}'.format(str(txt_avg_loss)))
        logger.info('The viz avg_loss is: {}'.format(str(viz_avg_loss)))

        # calculates
        logger.info('The sum of viz frozen part is {}'.format(str(sum_params(
            [p for p in trainer.viz_net.parameters() if not p.requires_grad]
        ))))

        logger.info('The sum of txt frozen part is {}'.format(str(sum_params(
            [p for p in trainer.txt_net.parameters() if not p.requires_grad]
        ))))

        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     no_improve_cnt = 0
        # else:
        #     no_improve_cnt += 1
        #     if no_improve_cnt > patience:
        #         logger.info("Already reach the patience")
                 
        #         sys.exit()

        if options.max_step is not None and step >= options.max_step:
            logger.info('Max-Step={} Quitting.'.format(options.max_step))
            sys.exit()

def get_train_dataset(options):
    viz_dataset = VisionDataset().initialize(options,
                                      vision_dir=options.train_path,
                                      filter_length=options.train_filter_length)

    text_dataset = ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)

    return combine_viz_text_dataset(viz_dataset, text_dataset)


def get_train_iterator(options, dataset):
    return make_combine_batch_iterator(options, dataset, shuffle=True,
            include_partial=True, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size)


def get_validation_dataset(options):
    viz_dataset =  VisionDataset().initialize(options,
                                      vision_dir=options.validation_path,
                                      filter_length=options.validation_filter_length)
    
    text_dataset = ReconstructDataset().initialize(options, text_path=options.validation_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.validation_data_type)

    return combine_viz_text_dataset(viz_dataset, text_dataset)

def get_validation_iterator(options, dataset):
    return make_combine_batch_iterator(options, dataset, shuffle=False,
            include_partial=True, filter_length=options.validation_filter_length,
            batch_size=options.validation_batch_size, length_to_size=options.length_to_size)


def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    return train_dataset, validation_dataset


def run(options):
    print(options)
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    bert_type = options.bert_type

    train_dataset, validation_dataset = get_train_and_validation(options)

    train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    txt_embeddings = train_dataset['embeddings']

    # get the vision embeddings
    viz_embeddings = get_model(options)

    if options.freeze_model:
        logger.info('The model has been frozen.')
        print('model: ', viz_embeddings)
        for p in viz_embeddings.parameters():
            p.requires_grad = False

    logger.info('Initializing model.')
    trainer = build_net(options, viz_embeddings, txt_embeddings, validation_iterator)
    logger.info('Vision Model:')
    for name, p in trainer.viz_net.named_parameters():
        logger.info('{} {} {}'.format(name, p.shape, p.requires_grad))

    logger.info('Text Model:')
    for name, p in trainer.txt_net.named_parameters():
        logger.info('{} {} {}'.format(name, p.shape, p.requires_grad))

    if options.save_init:
        logger.info('Saving model (init).')
        trainer.save_model(os.path.join(options.experiment_path, 'model_init.pt'))

    run_train(options, train_iterator, trainer, validation_iterator)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), '..', 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=500, type=int)
    parser.add_argument('--save_distinct', default=100, type=int)
    parser.add_argument('--save_after', default=500, type=int)
    parser.add_argument('--save_init', action='store_true')

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('./data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('./data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('./data/glove/glove.6B.300d.txt'), type=str)

    # Data (synthetic).
    parser.add_argument('--synthetic-nexamples', default=1000, type=int)
    parser.add_argument('--synthetic-vocabsize', default=1000, type=int)
    parser.add_argument('--synthetic-embeddingsize', default=1024, type=int)
    parser.add_argument('--synthetic-minlen', default=20, type=int)
    parser.add_argument('--synthetic-maxlen', default=21, type=int)
    parser.add_argument('--synthetic-seed', default=11, type=int)
    parser.add_argument('--synthetic-length', default=None, type=int)
    parser.add_argument('--use-synthetic-embeddings', action='store_true')

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=0, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Model.
    parser.add_argument('--arch', default='treelstm', choices=('treelstm', 'mlp', 'mlp-shared'))
    parser.add_argument('--hidden_dim', default=10, type=int)
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='margin', choices=('margin', 'softmax'))

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'elmo', 'bert', 'resnet', 'both', 'resnet18', 'resnet50', 'combine', 'combine50'))

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=3, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default=None, type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Training.
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)

    # Parsing.
    parser.add_argument('--postprocess', action='store_true')

    # Optimization.
    parser.add_argument('--lr', default=4e-3, type=float)

    # Add Optimization
    parser.add_argument('--word2idx', default=None, type=str)

    # add bert type
    parser.add_argument('--bert_type', default='bert-base-uncased', type=str)

    # bert
    parser.add_argument('--tokenizer_loading_path', default=None, type=str)
    parser.add_argument('--bertmodel_loading_path', default=None, type=str)

    # mask or not
    parser.add_argument('--mask', default=False, type=bool)

    # add vision type
    parser.add_argument('--vision_type', default='chair', type=str)
    # add vision model
    parser.add_argument('--vision_pretrain_path', default=None, type=str)
    # freeze model
    parser.add_argument('--freeze_model', default=1, type=int)

    # level attn
    parser.add_argument('--level_attn', default=0, type=int)

    # share diora_for_both text and 
    parser.add_argument('--diora_shared', default=0, type=int)

    # level or not
    parser.add_argument('--mixture', default=0, type=int)

    # txt2img
    parser.add_argument('--txt2img', default=0, type=int)

    # outside_pass
    parser.add_argument('--outside_attn', default=0, type=int)
    
    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.experiment_name:
        options.experiment_name = '{}'.format(options.uuid[:8])

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    if options.length_to_size is not None:
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        flags_to_use = []
        flags_to_use += ['arch']
        flags_to_use += ['compress']
        flags_to_use += ['emb']
        flags_to_use += ['hidden_dim']
        flags_to_use += ['normalize']
        flags_to_use += ['reconstruct_mode']

        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        options = init_with_flags_file(options, options.flags)

    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
