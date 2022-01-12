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

import json
from torchvision import transforms
import PIL.Image as Image
from PIL import ImageOps

from diora.data.dataloader import FixedLengthBatchSampler, VizDataset

import os

def get_config(config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    return config

class VisionDataset(object):
    def initialize(self, options, vision_dir, filter_length=0):
        
        vision_type = options.vision_type
        if vision_type == 'chair':
            parts = {'chair_head': 0, 'back_surface': 1, 'back_frame_vertical_bar': 2, 'back_frame_horizontal_bar': 3,  'chair_seat': 4, 'chair_arm': 5, 'arm_sofa_style': 6, 'arm_near_vertical_bar': 7,'arm_horizontal_bar': 8, 'central_support': 9, 'leg': 10, 'leg_bar': 11, 'pedestal': 12}
        elif vision_type == 'table':
            parts = {'tabletop': 0, 'drawer': 1, 'cabinet_door': 2, 'side_panel': 3, 'bottom_panel': 4, 'leg': 5, 'leg_bar': 6, 'central_support': 7, 'pedestal': 8, 'shelf': 9}
        elif vision_type == 'bed':
            parts = {'headboard': 0, 'bed_sleep_area': 1, 'bed_frame_horizontal_surface':2, 'bed_side_surface_panel': 3, 'bed_post': 4, 'leg': 5, 'surface_base': 6, 'ladder':7}
        elif vision_type == 'bag':
            parts = {'bag_body': 0, 'handle': 1, 'shoulder_strap': 2}
        else:
            raise Exception("The vision type {} is not valid.".format(vision_type))
        print('parts: ', parts)
        dir_list = [x for x in os.listdir(vision_dir) if not x.startswith('.') and '.' not in x]
        
        ids = []
        samples = []
        extra = {}
        example_ids = []

        for di in dir_list:
            sample = {}

            image_each = []
            target_each = []

            with open(os.path.join(vision_dir, di, 'idx2cat.json')) as f:
                part_dict = json.load(f)

            for val, t in part_dict.items():
                file_name = val + 'occluded.png'
                if not os.path.exists(os.path.join(vision_dir, di, file_name)):
                    continue
                
                image_each.append(os.path.join(vision_dir, di, file_name))
                target_each.append(parts[t])
            
            image_text_file = os.path.join(vision_dir, di, 'image_text.txt')
            if os.path.exists(image_text_file):
                with open(image_text_file, 'r') as f:
                    image_text = f.read()
                    image_text = image_text[:-1].strip().split()
            else:
                image_text = []

            vis_span_file = os.path.join(vision_dir, di, 'vis_spans.txt')
            if os.path.exists(vis_span_file):
                with open(vis_span_file) as f:
                    span = f.read()
                    image_span = json.loads(span)
            else:
                image_span = []
                    
            
            sample['images'] = image_each
            sample['targets'] = target_each
            sample['image_span'] = image_span
            sample['image_text'] = image_text
            
            samples.append(sample)
            ids.append(vision_type + '-' + di)
            example_ids.append(di)

        extra['example_ids'] = example_ids
        
        return {
            'samples': samples,
            'ids': ids,
            'extra': extra,
        }
        

def make_viz_batch_iterator(options, dset, shuffle=True, include_partial=False, filter_length=0,
                            batch_size=None, length_to_size=None):

    samples = dset['samples']
    ids = dset['ids']
    extra = dset['extra']

    cuda = options.cuda
    multigpu = options.multigpu
    ngpus = 1

    if cuda and multigpu:
        ngpus = torch.cuda.device_count()

    if options.reconstruct_mode in ('margin', 'softmax'):
        batch_iterator = VizBatchIterator(
            samples=samples, extra=extra, shuffle=shuffle, include_partial=include_partial,
            filter_length=filter_length, batch_size=batch_size, rank=options.local_rank,
            cuda=cuda, ngpus=ngpus, negative_sampler=None,
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
    )

    return default_config

class VizBatchIterator(object):
    
    def __init__(self, samples, extra, **kwargs):
        self.samples = samples
        self.extra = extra
        self.config = config = get_config(get_default_config(), **kwargs)
        self.loader = None
    
    def get_dataset_size(self):
        return len(self.samples)
    
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

        def collate_fn(batch):
            index, samples = zip(*batch)

            images = [sample['images'] for sample in samples]
            targets = [sample['targets'] for sample in samples]
            
            batch_map = {}
            batch_map['index'] = index
            batch_map['samples'] = torch.stack(images, dim=0)
            batch_map['targets'] = torch.stack(targets, dim=0)

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]
            
            return batch_map
        
        if self.loader is None:
            rng = np.random.RandomState(seed=random_seed)
            dataset = VizDataset(self.samples)
            sampler = FixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
                maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size,
                emb_type='viz')
            
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=workers, pin_memory=pin_memory, batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader
        
        def myiterator():
            for batch in self.loader:
                index = batch['index']
                samples = batch['samples']
                targets = batch['targets']

                batch_size, length, _, _ = samples.shape

                # neg_samples = None
                # if negative_sampler is not None:
                #     neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

                if cuda:
                    samples = samples.cuda()

                batch_map = {}
                batch_map['samples'] = samples
                batch_map['targets'] = targets
                # batch_map['neg_samples'] = neg_samples
                batch_map['batch_size'] = batch_size
                batch_map['length'] = length

                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()

        
        
        

            

