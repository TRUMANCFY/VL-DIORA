import json

import torch

from diora.logging.configuration import get_logger


class ParsePredictor(object):
    def __init__(self, options, net, word2idx, emb_type=None):
        super(ParsePredictor, self).__init__()
        self.options = options
        self.net = net
        self.word2idx = word2idx
        if word2idx is not None:
            self.idx2word = {v: k for k, v in word2idx.items()}
        else:
            self.idx2word = None
        self.logger = get_logger()
        self.emb_type = emb_type

    def parse_batch(self, batch_map):
        if self.options.emb == 'elmo' or self.emb_type == 'elmo':
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]
        elif self.options.emb == 'bert' or self.emb_type == 'bert':
            token_mask = batch_map['sentences']['token_mask']
            batch_size = token_mask.shape[0]
            length = token_mask.shape[1]
        elif self.options.emb.startswith('resnet') or self.emb_type == 'resnet' or self.emb_type == 'viz':
            samples = batch_map['samples']
            batch_size = samples.shape[0]
            length = samples.shape[1]
        else:
            raise Exception('The option emb type {} is not valid'.format(self.options.emb))
        
        # saved_scalar: {level: {position: (batch_size x level x 1)}}
        scalars = self.net.saved_scalars
        device = self.net.device
        dtype = torch.float32

        # Assign missing scalars, because there is no level 0
        for i in range(length):
            scalars[0][i] = torch.full((batch_size, 1), 1, dtype=dtype, device=device)

        trees = self.batched_cky(batch_map, scalars)

        return trees

    def batched_cky(self, batch_map, scalars):
        if self.options.emb == 'elmo' or self.emb_type == 'elmo':
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]
        elif self.options.emb == 'bert' or self.emb_type == 'bert':
            token_mask = batch_map['sentences']['token_mask']
            batch_size = token_mask.shape[0]
            length = token_mask.shape[1]
        elif self.options.emb.startswith('resnet') or self.emb_type == 'resnet' or self.emb_type == 'viz':
            samples = batch_map['samples']
            batch_size = samples.shape[0]
            length = samples.shape[1]
        else:
            raise Exception('The option emb type {} is not valid.'.format(self.options.emb))

        device = self.net.device
        dtype = torch.float32

        # Chart.
        chart = [torch.full((length-i, batch_size), 1, dtype=dtype, device=device) for i in range(length)]

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Assumes that the bottom-left most leaf is in the first constituent.
                # (batch_size, level, 1) N = level
                try:
                    spbatch = scalars[level][pos]
                except:
                    print('scalars: ', scalars)
                    print('level: ', level)
                    print('pos: ', pos)

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    assert l_level >= 0
                    assert l_pos >= 0
                    assert r_level >= 0
                    assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

                    lp = chart[l_level][l_pos].view(-1, 1)
                    rp = chart[r_level][r_pos].view(-1, 1)
                    sp = spbatch[:, idx].view(-1, 1)

                    lps.append(lp)
                    rps.append(rp)
                    sps.append(sp)

                    pairs.append((l, r))

                lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)

                ps = lps + rps + sps
                argmax = ps.argmax(1).long()

                valmax = ps[range(batch_size), argmax]
                chart[level][pos, :] = valmax

                for i, ix in enumerate(argmax.tolist()):
                    bp[i][level][pos] = pairs[ix]

        trees = []
        for i in range(batch_size):
            tree = self.follow_backpointers(bp[i], bp[i][-1][0])
            trees.append(tree)

        return trees

    def follow_backpointers(self, bp, pair):
        if isinstance(pair, int):
            return pair

        l, r = pair
        lout = self.follow_backpointers(bp, bp[l[0]][l[1]])
        rout = self.follow_backpointers(bp, bp[r[0]][r[1]])

        return (lout, rout)

