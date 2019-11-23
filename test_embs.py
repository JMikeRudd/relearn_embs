import torch
import argparse
import os
from ft_utils import *

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_embs(src_mod_name=None, tgt_mod_name=None, src_lang=None, tgt_lang=None, dict_inds=[0, 10000]):
    src_dir = os.path.join('trained_models', src_mod_name)
    tgt_dir = os.path.join('trained_models', tgt_mod_name)

    print(dict_inds, type(dict_inds[0]), type(dict_inds[1]), dict_inds[0]<dict_inds[1])
    assert isinstance(dict_inds, list) and len(dict_inds) == 2
    assert isinstance(dict_inds[0], int) and isinstance(dict_inds[1], int) and dict_inds[0] < dict_inds[1]
    dict_lb, dict_ub = dict_inds[0], dict_inds[1]

    srcmod = torch.load(os.path.join(src_dir, 'isometric_embedding')).to(device)
    tgtmod = torch.load(os.path.join(tgt_dir, 'isometric_embedding')).to(device)

    # Get old and new vectors
    max_voc = srcmod.metric.ft_model.max_vocab
    src_old, tgt_old = srcmod.metric.ft_model.vectors.to(device), tgtmod.metric.ft_model.vectors.to(device)
    src_new, tgt_new = srcmod.embed(torch.arange(max_voc).to(device)), tgtmod.embed(torch.arange(max_voc).to(device))

    torch.save(src_new.detach().cpu(), os.path.join(src_dir, 'vecs'))
    torch.save(tgt_new.detach().cpu(), os.path.join(tgt_dir, 'vecs'))
    src_new = torch.load('trained_models/en450').to(device)
    tgt_new = torch.load('trained_models/fr450').to(device)

    # Normalize old vectors
    src_old /= src_old.norm(dim=1, keepdim=True)
    src_old -= src_old.mean(0, keepdim=True)
    src_old /= src_old.norm(dim=1, keepdim=True)
    tgt_old /= tgt_old.norm(dim=1, keepdim=True)
    tgt_old -= tgt_old.mean(0, keepdim=True)
    tgt_old /= tgt_old.norm(dim=1, keepdim=True)

    # Get Dictionaries
    s2t = load_muse_dictionary('fasttext/{}-{}.txt'.format(src_lang, tgt_lang), srcmod.metric.ft_model.vocab.stoi, tgtmod.metric.ft_model.vocab.stoi).to(device)[dict_lb:dict_ub]
    t2s = load_muse_dictionary('fasttext/{}-{}.txt'.format(tgt_lang, src_lang), tgtmod.metric.ft_model.vocab.stoi, srcmod.metric.ft_model.vocab.stoi).to(device)[dict_lb:dict_ub]

    # Get optimal mappings
    old_dim, new_dim = srcmod.metric.ft_model.dim, srcmod.emb_space.emb_dim
    old_s2t_map = torch.nn.Linear(old_dim, old_dim, bias=False).to(device)
    old_t2s_map = torch.nn.Linear(old_dim, old_dim, bias=False).to(device)
    new_s2t_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)
    new_t2s_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)


    old_s2t_map.weight.data = procrustes(src_old, tgt_old, old_s2t_map.weight.data, s2t)
    old_t2s_map.weight.data = procrustes(tgt_old, src_old, old_t2s_map.weight.data, t2s)
    new_s2t_map.weight.data = procrustes(src_new, tgt_new, new_s2t_map.weight.data, s2t)
    new_t2s_map.weight.data = procrustes(tgt_new, src_new, new_t2s_map.weight.data, t2s)

    # Evaluate optimal mappings
    old_s2t, old_t2s, _, _ = evaluate_uwt(old_s2t_map, old_t2s_map, s2t, t2s, src_old, tgt_old)
    new_s2t, new_t2s, _, _ = evaluate_uwt(new_s2t_map, new_t2s_map, s2t, t2s, src_new, tgt_new)

    print('S2T:')
    print('\tcosine:\t{}\t{}\n\tcsls:\t{}\t{}\n'.format(old_s2t['cosine']['score'], new_s2t['cosine']['score'], old_s2t['csls']['score'], new_s2t['csls']['score']))
    print('T2S:')
    print('\tcosine:\t{}\t{}\n\tcsls:\t{}\t{}\n'.format(old_t2s['cosine']['score'], new_t2s['cosine']['score'], old_t2s['csls']['score'], new_t2s['csls']['score']))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_mod_name', type=str, default=None)
    parser.add_argument('--tgt_mod_name', type=str, default=None)
    parser.add_argument('--src_lang', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--dict_inds', type=int, nargs=2, default=[0, 10000])

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    test_embs(**args)

