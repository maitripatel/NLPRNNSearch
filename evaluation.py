from __future__ import print_function
import argparse
import time
import os
import sys
import subprocess
import tempfile
import pickle

import torch

import torch.utils.data
import itertools



import model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction



def convert_data(batch, vocab, device, reverse=False, unk=None, pad=None, sos=None, eos=None):
    max_len = max(len(x) for x in batch)
    padded = []
    for x in batch:
        if reverse:
            padded.append(
                ([] if eos is None else [eos]) +
                list(x[::-1]) +
                ([] if sos is None else [sos]))
        else:
            padded.append(
                ([] if sos is None else [sos]) +
                list(x) +
                ([] if eos is None else [eos]))
        padded[-1] = padded[-1] + [pad] * max(0, max_len - len(x))
        padded[-1] = list(map(lambda v: vocab['stoi'][v] if v in vocab['stoi'] else vocab['stoi'][unk], padded[-1]))
    padded = torch.LongTensor(padded).to(device)
    mask = padded.ne(vocab['stoi'][pad]).float()
    return padded, mask


def convert_str(batch, vocab):
    output = []
    for x in batch:
        output.append(list(map(lambda v: vocab['itos'][v], x)))
    return output


def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.items():
        v[idx] = k
    return v


def load_vocab(path):
    vocab = pickle.load(open(path, 'rb'))
    return vocab


def sort_batch(batch):
    batch = zip(*batch)
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    batch = zip(*batch)
    return list(batch)

class dataset(torch.utils.data.Dataset):
    def __init__(self, p_src, p_trg, src_max_len=None, trg_max_len=None):
        p_list = [p_src]
        if isinstance(p_trg, str):
            p_list.append(p_trg)
        else:
            p_list.extend(p_trg)
        lines = []
        for p in p_list:
            with open(p, encoding="utf8") as f:
                lines.append(f.readlines())
        assert len(lines[0]) == len(lines[1])
        self.data = []
        for line in itertools.zip_longest(*lines):
            line = list(map(lambda v: v.lower().strip(), line))
            if not any(line):
                continue
            line = list(map(lambda v: v.split(), line))
            if (src_max_len and len(line[0]) > src_max_len) \
                    or (trg_max_len and len(line[1]) > trg_max_len):
                continue
            self.data.append(line)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing Attention-based Neural Machine Translation Model')
    # data
    parser.add_argument('--src_vocab', type=str, help='source vocabulary')
    parser.add_argument('--trg_vocab', type=str, help='target vocabulary')
    parser.add_argument('--src_max_len', type=int, default=50, help='maximum length of source')
    parser.add_argument('--trg_max_len', type=int, default=50, help='maximum length of target')
    parser.add_argument('--test_src', type=str, help='source for testing')
    parser.add_argument('--test_trg', type=str, nargs='+', help='reference for testing')
    parser.add_argument('--eval_script', type=str, help='script for validation')
    # model
    parser.add_argument('--model', type=str, help='name of model')
    parser.add_argument('--name', type=str, help='name of checkpoint')
    parser.add_argument('--enc_ninp', type=int, default=550, help='size of source word embedding')
    parser.add_argument('--dec_ninp', type=int, default=550, help='size of target word embedding')
    parser.add_argument('--enc_nhid', type=int, default=1000, help='number of source hidden layer')
    parser.add_argument('--dec_nhid', type=int, default=1000, help='number of target hidden layer')
    parser.add_argument('--dec_natt', type=int, default=1000, help='number of target attention layer')
    parser.add_argument('--nreadout', type=int, default=550, help='number of maxout layer')
    parser.add_argument('--enc_emb_dropout', type=float, default=0.3, help='dropout rate for encoder embedding')
    parser.add_argument('--dec_emb_dropout', type=float, default=0.3, help='dropout rate for decoder embedding')
    parser.add_argument('--enc_hid_dropout', type=float, default=0.3, help='dropout rate for encoder hidden state')
    parser.add_argument('--readout_dropout', type=float, default=0.3, help='dropout rate for encoder hidden state')
    # search
    parser.add_argument('--beam_size', type=int, default=10, help='size of beam')
    # bookkeeping
    parser.add_argument('--seed', type=int, default=123, help='random number seed')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument('--save', type=str, default='./generation/', help='path to save generated sequence')
    # GPU
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    # Misc
    parser.add_argument('--info', type=str, help='info of the model')
    
    opt = parser.parse_args()
    
    # set the random seed manually
    torch.manual_seed(opt.seed)
    
    opt.cuda = opt.cuda and torch.cuda.is_available()
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    
    # load vocabulary for source and target
    src_vocab, trg_vocab = {}, {}
    src_vocab['stoi'] = load_vocab(opt.src_vocab)
    trg_vocab['stoi'] = load_vocab(opt.trg_vocab)
    src_vocab['itos'] = invert_vocab(src_vocab['stoi'])
    trg_vocab['itos'] = invert_vocab(trg_vocab['stoi'])
    UNK = '<unk>'
    SOS = '<sos>'
    EOS = '<eos>'
    PAD = '<pad>'
    opt.enc_pad = src_vocab['stoi'][PAD]
    opt.dec_sos = trg_vocab['stoi'][SOS]
    opt.dec_eos = trg_vocab['stoi'][EOS]
    opt.dec_pad = trg_vocab['stoi'][PAD]
    opt.enc_ntok = len(src_vocab['stoi'])
    opt.dec_ntok = len(trg_vocab['stoi'])
    
    # load dataset for testing
    test_dataset = dataset(opt.test_src, opt.test_trg)
    test_iter = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, collate_fn=lambda x: zip(*x))
    
    # create the model
    model = getattr(model, opt.model)(opt).to(device)
    
    state_dict = torch.load(os.path.join(opt.checkpoint, opt.name))
    model.load_state_dict(state_dict)
    model.eval()
    
    
    def bleu_script(f):
        ref_stem = opt.test_trg[0][:-1] + '*'
        cmd = '{eval_script} {refs} {hyp}'.format(eval_script=opt.eval_script, refs=ref_stem, hyp=f)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode > 0:
            sys.stderr.write(err)
            sys.exit(1)
        bleu = float(out)
        return bleu
    
    
    hyp_list = []
    ref_list = []
    start_time = time.time()
    for ix, batch in enumerate(test_iter, start=1):
        src_raw = batch[0]
        trg_raw = batch[1:]
        src, src_mask = convert_data(src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS)
        with torch.no_grad():
            output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
            best_hyp, best_score = output[0]
            best_hyp = convert_str([best_hyp], trg_vocab)
            hyp_list.append(best_hyp[0])
            ref = list(map(lambda x: x[0], trg_raw))
            ref_list.append(ref)
        print(ix, len(test_iter), 100. * ix / len(test_iter))
    elapsed = time.time() - start_time
    bleu1 = corpus_bleu(ref_list, hyp_list, smoothing_function=SmoothingFunction().method1)
    hyp_list = list(map(lambda x: ' '.join(x), hyp_list))
    p_tmp = tempfile.mktemp()
    f_tmp = open(p_tmp, 'w', encoding="utf8")
    f_tmp.write('\n'.join(hyp_list))
    f_tmp.close()
    bleu2 = bleu_script(p_tmp)
    print('BLEU score for model {} is {}/{}, {}'.format(opt.name, bleu1, bleu2, elapsed))
