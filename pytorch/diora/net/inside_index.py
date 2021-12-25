import torch
from diora.net.offset_cache import get_offset_cache


class InsideIndex(object):
    def get_pairs(self, level, i):
        pairs = []
        for constituent_num in range(0, level):
            l_level = constituent_num
            l_i = i - level + constituent_num
            r_level = level - 1 - constituent_num
            r_i = i
            pair = ((l_level, l_i), (r_level, r_i))
            pairs.append(pair)
        # the number of returning pair is level
        return pairs

    def get_all_pairs(self, level, n):
        # n is length
        pairs = []
        for i in range(level, n):
            pairs += self.get_pairs(level, i)
        return pairs


class InsideIndexCheck(object):
    def __init__(self, length, spans, siblings):
        sib_map = {}
        for x, y, n in siblings:
            sib_map[x] = (y, n)
            sib_map[y] = (x, n)

        check = {}
        for sibling, (target, name) in sib_map.items():
            xlength = target[1] - target[0]
            xlevel = xlength - 1
            xpos = target[0]
            tgt = (xlevel, xpos)

            slength = sibling[1] - sibling[0]
            slevel = slength - 1
            spos = sibling[0]
            sis = (slevel, spos)

            check[(tgt, sis)] = True
        self.check = check

    def is_valid(self, tgt, sis):
        return (tgt, sis) in self.check


def get_inside_index(length, level, offset_cache=None, cuda=False):
    
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = InsideIndex()
    pairs = index.get_all_pairs(level, length)

    L = length - level
    n_constituents = len(pairs) // L
    idx_l, idx_r = [], []

    for i in range(n_constituents):
        index_l, index_r = [], []

        lvl_l = i
        lvl_r = level - i - 1
        lstart, lend = 0, L
        rstart, rend = length - L - lvl_r, length - lvl_r

        if lvl_l < 0:
            lvl_l = length + lvl_l
        if lvl_r < 0:
            lvl_r = length + lvl_r

        for pos in range(lstart, lend):
            offset = offset_cache[lvl_l]
            idx = offset + pos
            index_l.append(idx)

        for pos in range(rstart, rend):
            offset = offset_cache[lvl_r]
            idx = offset + pos
            index_r.append(idx)

        idx_l.append(index_l)
        idx_r.append(index_r)

    device = torch.cuda.current_device() if cuda else None
    idx_l = torch.tensor(idx_l, dtype=torch.int64, device=device
            ).transpose(0, 1).contiguous().flatten()
    idx_r = torch.tensor(idx_r, dtype=torch.int64, device=device
            ).transpose(0, 1).contiguous().flatten()
    
    # print('======')
    # print('level: ', level)
    # print('length: ', length)
    # print('offset_cache: ', offset_cache)
    # print('idx_l: ', idx_l)
    # print('idx_r: ', idx_r)    

    return idx_l, idx_r

# ======
# level:  1
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='cuda:1')
# idx_r:  tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], device='cuda:1')
# ======
# level:  2
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11,  1, 12,  2, 13,  3, 14,  4, 15,  5, 16,  6, 17,  7, 18,  8, 19],
#        device='cuda:1')
# idx_r:  tensor([12,  2, 13,  3, 14,  4, 15,  5, 16,  6, 17,  7, 18,  8, 19,  9, 20, 10],
#        device='cuda:1')
# ======
# level:  3
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21,  1, 12, 22,  2, 13, 23,  3, 14, 24,  4, 15, 25,  5, 16, 26,
#          6, 17, 27,  7, 18, 28], device='cuda:1')
# idx_r:  tensor([22, 13,  3, 23, 14,  4, 24, 15,  5, 25, 16,  6, 26, 17,  7, 27, 18,  8,
#         28, 19,  9, 29, 20, 10], device='cuda:1')
# ======
# level:  4
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30,  1, 12, 22, 31,  2, 13, 23, 32,  3, 14, 24, 33,  4, 15,
#         25, 34,  5, 16, 26, 35,  6, 17, 27, 36], device='cuda:1')
# idx_r:  tensor([31, 23, 14,  4, 32, 24, 15,  5, 33, 25, 16,  6, 34, 26, 17,  7, 35, 27,
#         18,  8, 36, 28, 19,  9, 37, 29, 20, 10], device='cuda:1')
# ======
# level:  5
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38,  1, 12, 22, 31, 39,  2, 13, 23, 32, 40,  3, 14, 24,
#         33, 41,  4, 15, 25, 34, 42,  5, 16, 26, 35, 43], device='cuda:1')
# idx_r:  tensor([39, 32, 24, 15,  5, 40, 33, 25, 16,  6, 41, 34, 26, 17,  7, 42, 35, 27,
#         18,  8, 43, 36, 28, 19,  9, 44, 37, 29, 20, 10], device='cuda:1')
# ======
# level:  6
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45,  1, 12, 22, 31, 39, 46,  2, 13, 23, 32, 40, 47,
#          3, 14, 24, 33, 41, 48,  4, 15, 25, 34, 42, 49], device='cuda:1')
# idx_r:  tensor([46, 40, 33, 25, 16,  6, 47, 41, 34, 26, 17,  7, 48, 42, 35, 27, 18,  8,
#         49, 43, 36, 28, 19,  9, 50, 44, 37, 29, 20, 10], device='cuda:1')
# ======
# level:  7
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51,  1, 12, 22, 31, 39, 46, 52,  2, 13, 23, 32,
#         40, 47, 53,  3, 14, 24, 33, 41, 48, 54], device='cuda:1')
# idx_r:  tensor([52, 47, 41, 34, 26, 17,  7, 53, 48, 42, 35, 27, 18,  8, 54, 49, 43, 36,
#         28, 19,  9, 55, 50, 44, 37, 29, 20, 10], device='cuda:1')
# ======
# level:  8
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51, 56,  1, 12, 22, 31, 39, 46, 52, 57,  2, 13,
#         23, 32, 40, 47, 53, 58], device='cuda:1')
# idx_r:  tensor([57, 53, 48, 42, 35, 27, 18,  8, 58, 54, 49, 43, 36, 28, 19,  9, 59, 55,
#         50, 44, 37, 29, 20, 10], device='cuda:1')
# ======
# level:  9
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51, 56, 60,  1, 12, 22, 31, 39, 46, 52, 57, 61],
#        device='cuda:1')
# idx_r:  tensor([61, 58, 54, 49, 43, 36, 28, 19,  9, 62, 59, 55, 50, 44, 37, 29, 20, 10],
#        device='cuda:1')
# ======
# level:  10
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51, 56, 60, 63], device='cuda:1')
# idx_r:  tensor([64, 62, 59, 55, 50, 44, 37, 29, 20, 10], device='cuda:1')
# ======
# level:  1
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='cuda:0')
# idx_r:  tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], device='cuda:0')
# ======
# level:  2
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11,  1, 12,  2, 13,  3, 14,  4, 15,  5, 16,  6, 17,  7, 18,  8, 19],
#        device='cuda:0')
# idx_r:  tensor([12,  2, 13,  3, 14,  4, 15,  5, 16,  6, 17,  7, 18,  8, 19,  9, 20, 10],
#        device='cuda:0')
# ======
# level:  3
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21,  1, 12, 22,  2, 13, 23,  3, 14, 24,  4, 15, 25,  5, 16, 26,
#          6, 17, 27,  7, 18, 28], device='cuda:0')
# idx_r:  tensor([22, 13,  3, 23, 14,  4, 24, 15,  5, 25, 16,  6, 26, 17,  7, 27, 18,  8,
#         28, 19,  9, 29, 20, 10], device='cuda:0')
# ======
# level:  4
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30,  1, 12, 22, 31,  2, 13, 23, 32,  3, 14, 24, 33,  4, 15,
#         25, 34,  5, 16, 26, 35,  6, 17, 27, 36], device='cuda:0')
# idx_r:  tensor([31, 23, 14,  4, 32, 24, 15,  5, 33, 25, 16,  6, 34, 26, 17,  7, 35, 27,
#         18,  8, 36, 28, 19,  9, 37, 29, 20, 10], device='cuda:0')
# ======
# level:  5
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38,  1, 12, 22, 31, 39,  2, 13, 23, 32, 40,  3, 14, 24,
#         33, 41,  4, 15, 25, 34, 42,  5, 16, 26, 35, 43], device='cuda:0')
# idx_r:  tensor([39, 32, 24, 15,  5, 40, 33, 25, 16,  6, 41, 34, 26, 17,  7, 42, 35, 27,
#         18,  8, 43, 36, 28, 19,  9, 44, 37, 29, 20, 10], device='cuda:0')
# ======
# level:  6
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45,  1, 12, 22, 31, 39, 46,  2, 13, 23, 32, 40, 47,
#          3, 14, 24, 33, 41, 48,  4, 15, 25, 34, 42, 49], device='cuda:0')
# idx_r:  tensor([46, 40, 33, 25, 16,  6, 47, 41, 34, 26, 17,  7, 48, 42, 35, 27, 18,  8,
#         49, 43, 36, 28, 19,  9, 50, 44, 37, 29, 20, 10], device='cuda:0')
# ======
# level:  7
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51,  1, 12, 22, 31, 39, 46, 52,  2, 13, 23, 32,
#         40, 47, 53,  3, 14, 24, 33, 41, 48, 54], device='cuda:0')
# idx_r:  tensor([52, 47, 41, 34, 26, 17,  7, 53, 48, 42, 35, 27, 18,  8, 54, 49, 43, 36,
#         28, 19,  9, 55, 50, 44, 37, 29, 20, 10], device='cuda:0')
# ======
# level:  8
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51, 56,  1, 12, 22, 31, 39, 46, 52, 57,  2, 13,
#         23, 32, 40, 47, 53, 58], device='cuda:0')
# idx_r:  tensor([57, 53, 48, 42, 35, 27, 18,  8, 58, 54, 49, 43, 36, 28, 19,  9, 59, 55,
#         50, 44, 37, 29, 20, 10], device='cuda:0')
# ======
# level:  9
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51, 56, 60,  1, 12, 22, 31, 39, 46, 52, 57, 61],
#        device='cuda:0')
# idx_r:  tensor([61, 58, 54, 49, 43, 36, 28, 19,  9, 62, 59, 55, 50, 44, 37, 29, 20, 10],
#        device='cuda:0')
# ======
# level:  10
# length:  11
# offset_cache:  {0: 0, 1: 11, 2: 21, 3: 30, 4: 38, 5: 45, 6: 51, 7: 56, 8: 60, 9: 63, 10: 65}
# idx_l:  tensor([ 0, 11, 21, 30, 38, 45, 51, 56, 60, 63], device='cuda:0')
# idx_r:  tensor([64, 62, 59, 55, 50, 44, 37, 29, 20, 10], device='cuda:0')

