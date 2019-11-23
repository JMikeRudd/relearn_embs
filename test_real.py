import torch
import argparse
import os
from ft_utils import *

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_embs(src_mod_name=None, tgt_mod_name=None, src_lang=None, tgt_lang=None):
    src_dir = os.path.join('relearn_embs/trained_models', src_mod_name)
    tgt_dir = os.path.join('relearn_embs/trained_models', tgt_mod_name)
    assert os.path.exists(src_dir) and os.path.exists(tgt_dir)

    srcmod = torch.load(os.path.join(src_dir, 'isometric_embedding')).to(device)
    tgtmod = torch.load(os.path.join(tgt_dir, 'isometric_embedding')).to(device)

    translator = torch.load('vae-kl-div/trained_models/enfr/translator')

    # Get old and new vectors
    max_voc = srcmod.metric.ft_model.max_vocab
    src_old, tgt_old = srcmod.metric.ft_model.vectors.to(device), tgtmod.metric.ft_model.vectors.to(device)
    src_new, tgt_new = srcmod.embed(torch.arange(max_voc).to(device)), tgtmod.embed(torch.arange(max_voc).to(device))

    torch.save(src_new.detach().cpu(), os.path.join(src_dir, 'vecs'))
    torch.save(tgt_new.detach().cpu(), os.path.join(tgt_dir, 'vecs'))

    # Normalize old vectors
    src_old /= src_old.norm(dim=1, keepdim=True)
    src_old -= src_old.mean(0, keepdim=True)
    src_old /= src_old.norm(dim=1, keepdim=True)
    tgt_old /= tgt_old.norm(dim=1, keepdim=True)
    tgt_old -= tgt_old.mean(0, keepdim=True)
    tgt_old /= tgt_old.norm(dim=1, keepdim=True)

    # Get Dictionaries
    s2t = load_muse_dictionary('fasttext/{}-{}.5000-6500.txt'.format(src_lang, tgt_lang), srcmod.metric.ft_model.vocab.stoi, tgtmod.metric.ft_model.vocab.stoi).to(device)
    t2s = load_muse_dictionary('fasttext/{}-{}.5000-6500.txt'.format(tgt_lang, src_lang), tgtmod.metric.ft_model.vocab.stoi, srcmod.metric.ft_model.vocab.stoi).to(device)

    orig_s2t, orig_t2s, _, _ = translator.evaluate(s2t, t2s, modes=['cosine', 'csls'])
    print('Original Model, No Procrustes')
    print_scores(orig_s2t, orig_t2s)

    for _ in range(5):
        translator.procrustes(mode='joint')

    proc_s2t, proc_t2s, _, _ = translator.evaluate(s2t, t2s, modes=['cosine', 'csls'])
    print('\nOriginal Model, Procrustes')
    print_scores(proc_s2t, proc_t2s)

    # Get new dicos
    s2t_joint_dico = translator.joint_dico
    t2s_joint_dico = translator.joint_dico[:, [1, 0]]

    # Get optimal mappings
    new_dim = srcmod.emb_space.emb_dim
    new_s2t_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)
    new_t2s_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)

    new_s2t_map.weight.data = procrustes(src_new, tgt_new, new_s2t_map.weight.data, s2t_joint_dico)
    new_t2s_map.weight.data = procrustes(tgt_new, src_new, new_t2s_map.weight.data, t2s_joint_dico)

    # Evaluate optimal mappings
    new_s2t, new_t2s, _, _ = evaluate_uwt(new_s2t_map, new_t2s_map, s2t, t2s, src_new, tgt_new)

    print('\nNew Vectors, One Proc Update')
    print_scores(new_s2t, new_t2s)

def print_scores(s2t, t2s):
    print('S2T:')
    print('\tcosine: {}\n\tcsls: {}\n'.format(s2t['cosine']['score'], s2t['csls']['score']))
    print('T2S:')
    print('\tcosine: {}\n\tcsls: {}\n'.format(t2s['cosine']['score'], t2s['csls']['score']))



def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_mod_name', type=str, default=None)
    parser.add_argument('--tgt_mod_name', type=str, default=None)
    parser.add_argument('--src_lang', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    test_embs(**args)

