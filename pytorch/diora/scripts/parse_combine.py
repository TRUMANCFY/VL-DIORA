import os
import collections
import json
import types

import torch

from tqdm import tqdm

from train_combine import argument_parser, parse_args, configure
from train_combine import get_validation_dataset, get_validation_iterator
from train_combine import build_net

from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY

from diora.data.dataset import Vocabulary

from diora.net.vision_models import get_model

import numpy as np


punctuation_words = set([x.lower() for x in ['.', ',', ':', '-LRB-', '-RRB-', '\'\'',
    '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']])


def remove_using_flat_mask(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def postprocess(tr, tokens=None):
    if tokens is None:
        tokens = flatten_tree(tr)

    # Don't remove the last token. It's not punctuation.
    if tokens[-1].lower() not in punctuation_words:
        return tr

    mask = [True] * (len(tokens) - 1) + [False]
    tr, kept, removed = remove_using_flat_mask(tr, mask)
    assert len(kept) == len(tokens) - 1, 'Incorrect tokens left. Original = {}, Output = {}, Kept = {}'.format(
        binary_tree, tr, kept)
    assert len(kept) > 0, 'No tokens left. Original = {}'.format(tokens)
    assert len(removed) == 1
    tr = (tr, tokens[-1])

    return tr


def override_init_with_batch(var):
    init_with_batch = var.init_with_batch

    def func(self, *args, **kwargs):
        init_with_batch(*args, **kwargs)
        self.saved_scalars = {i: {} for i in range(self.length)}
        self.saved_scalars_out = {i: {} for i in range(self.length)}

    var.init_with_batch = types.MethodType(func, var)


def override_inside_hook(var):
    def func(self, level, h, c, s):
        length = self.length
        B = self.batch_size
        L = length - level

        # print('length: ', length)
        # print('level: ', level)
        # print('s.shape: ', s.shape)

        # N is level

        # s shape is (B, L, N, 1)
        assert s.shape[0] == B
        assert s.shape[1] == L
        # assert s.shape[2] == N
        assert s.shape[3] == 1
        assert len(s.shape) == 4
        smax = s.max(2, keepdim=True)[0]
        s = s - smax
        # print('update s.shape: ', s.shape)

        for pos in range(L):
            # print('level: ', level)
            # print('pos: ', pos)
            # print('s[:, pos, :]: ', s[:, pos, :].shape)
            self.saved_scalars[level][pos] = s[:, pos, :]

    var.inside_hook = types.MethodType(func, var)


def replace_leaves(tree, leaves):
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree += [newnode]

        return sofar, newtree

    _, newtree = func(tree)

    return newtree

def get_len(tree):
    if isinstance(tree, str) or isinstance(tree, int):
        return 1
    
    return sum([get_len(x) for x in tree])


def get_spans(tree):
    queue = [(tree, 0)]
    spans = []

    while queue:
        current_node = queue.pop(0)

        tree = current_node[0]
        offset = current_node[1]

        spans.append((offset, offset + get_len(tree) - 1))
        if not isinstance(tree[0], str) and not isinstance(tree[0], int):
            queue.append((tree[0], offset))
        
        if not isinstance(tree[1], str) and not isinstance(tree[1], int):
            queue.append((tree[1], offset + get_len(tree[0])))
        
    return set(spans)

def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0

    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    
    for span in span2:
        if span not in span1:
            fn += 1
    
    return tp, fp, fn


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    txt_embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}
    
    viz_embeddings = get_model(options)

    logger.info('Initializing model.')
    trainer = build_net(options, viz_embeddings, txt_embeddings, validation_iterator)

    # Parse txt
    txt_diora = trainer.txt_net.diora

    ## Monkey patch parsing specific methods.
    override_init_with_batch(txt_diora)
    override_inside_hook(txt_diora)

    ## Turn off outside pass.
    trainer.txt_net.diora.outside = False

    ## Eval mode.
    trainer.txt_net.eval()

    # Parse viz
    viz_diora = trainer.viz_net.diora
    override_init_with_batch(viz_diora)
    override_inside_hook(viz_diora)
    trainer.viz_net.diora.outside = False
    trainer.viz_net.eval()

    ## Parse predictor.
    viz_parse_predictor = CKY(options=options, net=viz_diora, word2idx=word2idx, emb_type='viz')
    txt_parse_predictor = CKY(options=options, net=txt_diora, word2idx=word2idx, emb_type='elmo')

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    txt_output_path = os.path.abspath(os.path.join(options.experiment_path, 'txt_parse.jsonl'))

    logger.info('Writing output to = {}'.format(txt_output_path))

    txt_f = open(txt_output_path, 'w')

    viz_output_path = os.path.abspath(os.path.join(options.experiment_path, 'viz_parse.jsonl'))

    logger.info('Writing output to = {}'.format(viz_output_path))

    viz_f = open(viz_output_path, 'w')

    pred_class_dict = {}

    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            samples = batch_map['samples']
            targets = batch_map['targets']
            batch_size = samples.shape[0]
            viz_length = samples.shape[1]

            sentences = batch_map['sentences']
            txt_length = samples.shape[1]

            # Skip very short sentences.
            if txt_length <= 2 or viz_length < 2:
                continue
            if options.txt2img:
                _ = trainer.step_txt(batch_map, train=False, compute_loss=False)

                pred_labels = trainer.get_class(batch_map)[1]
                
                txt_trees = txt_parse_predictor.parse_batch(batch_map)

                for ii, tr in enumerate(txt_trees):
                    example_id = batch_map['example_ids'][ii]
                    s = [idx2word[idx] for idx in sentences[ii].tolist()]
                    tr = replace_leaves(tr, s)
                    if options.postprocess:
                        tr = postprocess(tr, s)
                    o = collections.OrderedDict(example_id=example_id, tree=tr)
                    txt_f.write(json.dumps(o) + '\n')

                _ = trainer.step(batch_map, train=False, compute_loss=False)
                viz_trees = viz_parse_predictor.parse_batch(batch_map)

                for ii, tr in enumerate(viz_trees):
                    example_id = batch_map['example_ids'][ii]
                    # s = [idx2word[idx] for idx in sentences[ii].tolist()]
                    s = targets[ii].tolist()
                    pred_class_dict[example_id] = pred_labels[ii].tolist()

                    tr = replace_leaves(tr, s)
                    if options.postprocess:
                        tr = postprocess(tr, s)
                    o = collections.OrderedDict(example_id=example_id, tree=tr)

                    viz_f.write(json.dumps(o) + '\n')
            else:
                _ = trainer.step_viz(batch_map, train=False, compute_loss=False)
                
                pred_labels = trainer.get_class(batch_map)[1]

                viz_trees = viz_parse_predictor.parse_batch(batch_map)

                for ii, tr in enumerate(viz_trees):
                    example_id = batch_map['example_ids'][ii]
                    # s = [idx2word[idx] for idx in sentences[ii].tolist()]
                    s = targets[ii].tolist()
                    pred_class_dict[example_id] = pred_labels[ii].tolist()

                    tr = replace_leaves(tr, s)
                    if options.postprocess:
                        tr = postprocess(tr, s)
                    o = collections.OrderedDict(example_id=example_id, tree=tr)

                    viz_f.write(json.dumps(o) + '\n')
                
                _ = trainer.step(batch_map, train=False, compute_loss=False)

                pred_labels = trainer.get_class(batch_map)[1]
                
                txt_trees = txt_parse_predictor.parse_batch(batch_map)

                for ii, tr in enumerate(txt_trees):
                    example_id = batch_map['example_ids'][ii]
                    s = [idx2word[idx] for idx in sentences[ii].tolist()]
                    tr = replace_leaves(tr, s)
                    if options.postprocess:
                        tr = postprocess(tr, s)
                    o = collections.OrderedDict(example_id=example_id, tree=tr)
                    txt_f.write(json.dumps(o) + '\n')
                
    txt_f.close()
    viz_f.close()

    # evaluate text reuslt
    with open(txt_output_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    lines_res = [json.loads(x) for x in lines]

    sent_f1_txt, corpus_f1_txt = [], [0., 0., 0.]

    for idx, line in enumerate(lines_res):
        pred_txt = get_spans(line['tree'])
        example_id = line['example_id']

        with open(os.path.join(options.validation_path, example_id, 'lan_spans.txt'), 'r') as w:
            gold_txt = json.loads(w.read())

        print('line: ', line['tree'])
        print('pred_span: ', pred_txt)
        print('gold_txt: ', gold_txt)

        gold_txt = set([(a, b) for a, b in gold_txt])
        tp_txt, fp_txt, fn_txt = get_stats(pred_txt, gold_txt)
        corpus_f1_txt[0] += tp_txt
        corpus_f1_txt[1] += fp_txt
        corpus_f1_txt[2] += fn_txt

        overlap_txt = pred_txt.intersection(gold_txt)
        prec_txt = float(len(overlap_txt)) / (len(pred_txt) + 1e-8)
        reca_txt = float(len(overlap_txt)) / (len(gold_txt) + 1e-8)

        if len(gold_txt) == 0:
            reca_txt = 1.
            if len(pred_txt) == 0:
                pred_txt = 1.
        
        f1_txt = 2 * prec_txt * reca_txt / (prec_txt + reca_txt + 1e-8)
        sent_f1_txt.append(f1_txt)
    
    tp_txt, fp_txt, fn_txt = corpus_f1_txt
    prec_txt = tp_txt / (tp_txt + fp_txt)
    recall_txt = tp_txt / (tp_txt + fn_txt)
    corpus_f1_txt = 2 * prec_txt * recall_txt / (prec_txt + recall_txt + 1e-8)
    sent_f1_txt = np.mean(np.array(sent_f1_txt))

    print('prec_txt: ', prec_txt)
    print('recall_txt: ', recall_txt)
    print('corpus_f1_txt: ', corpus_f1_txt)
    print('sent_f1_txt: ', sent_f1_txt)

    

    # evaluate the result
    with open(viz_output_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    lines_res = [json.loads(x) for x in lines]

    sent_f1_txt, corpus_f1_txt = [], [0., 0., 0.]

    for idx, line in enumerate(lines_res):
        pred_txt = get_spans(line['tree'])
        example_id = line['example_id']

        with open(os.path.join(options.validation_path, example_id, 'vis_spans.txt'), 'r') as w:
            gold_txt = json.loads(w.read())
        
        print('example_id: ', example_id)
        print('class str: ', '|'.join([str(x) for x in pred_class_dict[example_id]]))
        print('line: ', line['tree'])
        print("gold_txt: ", gold_txt)
        
        gold_txt = set([(a, b) for a, b in gold_txt])
        tp_txt, fp_txt, fn_txt = get_stats(pred_txt, gold_txt)
        corpus_f1_txt[0] += tp_txt
        corpus_f1_txt[1] += fp_txt
        corpus_f1_txt[2] += fn_txt

        overlap_txt = pred_txt.intersection(gold_txt)
        prec_txt = float(len(overlap_txt)) / (len(pred_txt) + 1e-8)
        reca_txt = float(len(overlap_txt)) / (len(gold_txt) + 1e-8)

        if len(gold_txt) == 0:
            reca_txt = 1.
            if len(pred_txt) == 0:
                pred_txt = 1.
        
        f1_txt = 2 * prec_txt * reca_txt / (prec_txt + reca_txt + 1e-8)
        sent_f1_txt.append(f1_txt)
    
    tp_txt, fp_txt, fn_txt = corpus_f1_txt
    prec_txt = tp_txt / (tp_txt + fp_txt)
    recall_txt = tp_txt / (tp_txt + fn_txt)
    corpus_f1_txt = 2 * prec_txt * recall_txt / (prec_txt + recall_txt + 1e-8)
    sent_f1_txt = np.mean(np.array(sent_f1_txt))

    print('prec_viz: ', prec_txt)
    print('recall_viz: ', recall_txt)
    print('corpus_f1_viz: ', corpus_f1_txt)
    print('sent_f1_viz: ', sent_f1_txt)



if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
