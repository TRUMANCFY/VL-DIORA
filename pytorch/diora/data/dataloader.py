import os

import torch
from torch.utils.data import Sampler

import numpy as np

from diora.logging.configuration import get_logger

from torchvision import transforms
import PIL.Image as Image
from PIL import ImageOps

IMG_TRANSFORM = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0], [1])
                ])

class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None, maxlen=None,
                 length_to_size=None, emb_type='elmo'):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = { 0: self.batch_size }
        self.logger = get_logger()

        self.emb_type = emb_type

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size

        return batch_size

    def reset(self):
        """
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """

        # Record the lengths of each example.
        length_map = {}
        for i in range(len(self.data_source)):
            if self.emb_type == 'elmo':
                x = self.data_source.dataset[i]
                length = len(x)
            elif self.emb_type == 'bert':
                x = self.data_source.dataset[i]['origin_input_ids']
                length = x.shape[1]
            elif self.emb_type == 'viz':
                # x = self.data_source.dataset[i]
                length = len(self.data_source.dataset[i]['images'])
            elif self.emb_type == 'both':
                # length = 100 * len(se)
                img = self.data_source.viz_dataset.dataset[i]['images']
                sent = self.data_source.text_dataset.dataset[i]
                txt_len = len(sent)
                img_len = len(img)
                length = (txt_len, img_len)
            else:
                raise Exception('Embedding type is not valid.')

            if self.maxlen is not None and self.maxlen > 0:
                if isinstance(length, int) and length > self.maxlen:
                    continue
                elif isinstance(length, tuple) and length[0] > self.maxlen:
                    continue

            length_map.setdefault(length, []).append(i)

        # Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state = {}
        for length, arr in length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = nbatches * batch_size < len(arr)
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)


        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1

        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index
        
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)

class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item

    def __len__(self):
        return len(self.dataset)

class VizDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        # self.local_records = {}
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image_path_list = item['images']
        target_list = item['targets']

        # get the image
        images = []
        for path in image_path_list:
            img = Image.open(path).convert('L')
            img = ImageOps.invert(img)
            img = IMG_TRANSFORM(img)
            images.append(img)

        # combine images
        images = torch.stack(images).squeeze(1)

        new_item = {}
        new_item['images'] = images
        new_item['targets'] = torch.LongTensor(target_list)

        # if index in self.local_records:
        #     origin_img = self.local_records[index]
        #     assert torch.equal(origin_img, images), 'Images not consistent!'
        # else:
        #     self.local_records[index] = new_item['images']

        return index, new_item
        

    def __len__(self):
        return len(self.dataset)

class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, viz_data):
        assert len(text_data) == len(viz_data), 'The length of data should be the same.'
        self.text_dataset = SimpleDataset(text_data)
        self.viz_dataset = VizDataset(viz_data)
    
    def __getitem__(self, index):
        return index, self.text_dataset[index][1], self.viz_dataset[index][1]
    
    def __len__(self):
        return len(self.text_dataset)
