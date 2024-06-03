from time import time
import csv

import torch
from tqdm import tqdm

# from line_profiler import profile

from lps import DRYLogitsProcessor
from nlps import NewDRYLogitsProcessor

torch.manual_seed(0)

def test(context_length, nrepeats):
    # setup
    vocab_size = 32000
    input_ids = torch.arange(context_length).unsqueeze(0).to('cuda')
    scores = torch.randn(vocab_size, dtype=torch.float32).unsqueeze(0).to('cuda')
    set_repeats(input_ids, nrepeats)

    # prepare
    nlp = NewDRYLogitsProcessor(multiplier=1.1, base=1.1, allowed_length=0, sequence_breakers=set([]), _range=0)
    lp = DRYLogitsProcessor(multiplier=1.1, base=1.1, allowed_length=0, sequence_breakers=set([]), _range=0)
    assert lp(input_ids, scores.clone()).sum() == nlp(input_ids, scores.clone()).sum()

    # test
    n = 5000
    d1 = bench(n, input_ids, scores, f"Original (ctx={context_length}, repeats={nrepeats})", lp)
    d2 = bench(n, input_ids, scores, f"Improved (ctx={context_length}, repeats={nrepeats})", nlp)

    # log
    print(f'Speedup: {d1/d2:.2f}x\n')
    write_stats([context_length, nrepeats, d1, d2, n])


def set_repeats(input_ids, n):
    for i in range(n):
        input_ids[0][i] = input_ids[0][-1]

def bench(n, input_ids, scores, msg, cb):
    d = 0.0
    for _ in tqdm(range(n), total=n, leave=True, desc=f'Benchmarking {msg}'):
        # to ensure pristine values on the same device each time
        iter_ids, iter_scores = input_ids.clone(), scores.clone()
        t = time()
        _ = cb(iter_ids, iter_scores)
        d += time() - t
        del iter_ids
        del iter_scores
    return d


if __name__ == '__main__':
    context_lengths = [2048, 4096, 8192, 16384]
    nrepeats = [0, 1, 2, 4, 8, 16, 32, 64]

    with open('stats.csv', 'w') as f:
        write_stats = csv.writer(f).writerow
        write_stats(['context_length', 'nrepeats', 'original_secs', 'improved_secs', 'iterations'])
        for context_length, nrepeat in ((c,r) for r in nrepeats for c in context_lengths):
            test(context_length=context_length, nrepeats=nrepeat)
