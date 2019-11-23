import torch
from ft_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

enmod = torch.load('trained_models/en_300_reduced_weighted/isometric_embedding').to(device)
frmod = torch.load('trained_models/fr_300_reduced_weighted/isometric_embedding').to(device)

# Get old and new vectors
en_old, fr_old = enmod.metric.ft_model.vectors.to(device), frmod.metric.ft_model.vectors.to(device)
en_new, fr_new = enmod.embed(torch.arange(200000).to(device)), frmod.embed(torch.arange(200000).to(device))

enfr = load_muse_dictionary('fasttext/en-fr.txt', enmod.metric.ft_model.vocab.stoi, frmod.metric.ft_model.vocab.stoi).to(device)
fren = load_muse_dictionary('fasttext/fr-en.txt', frmod.metric.ft_model.vocab.stoi, enmod.metric.ft_model.vocab.stoi).to(device)

enfr_test = load_muse_dictionary('fasttext/en-fr.5000-6500.txt', enmod.metric.ft_model.vocab.stoi, frmod.metric.ft_model.vocab.stoi).to(device)
fren_test = load_muse_dictionary('fasttext/fr-en.5000-6500.txt', frmod.metric.ft_model.vocab.stoi, enmod.metric.ft_model.vocab.stoi).to(device)

real_dico_s2t = torch.load('trained_models/refined_enfr_dico')
real_dico_t2s = real_dico_s2t[:,[1,0]]

old_dim, new_dim = enmod.metric.ft_model.dim, enmod.emb_space.emb_dim
old_enfr_map = torch.nn.Linear(old_dim, old_dim, bias=False).to(device)
old_fren_map = torch.nn.Linear(old_dim, old_dim, bias=False).to(device)
new_enfr_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)
new_fren_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)


# Under Refined Dico
old_enfr_map.weight.data = procrustes(en_old, fr_old, old_enfr_map.weight.data, real_dico_s2t)
old_fren_map.weight.data = procrustes(fr_old, en_old, old_fren_map.weight.data, real_dico_t2s)
new_enfr_map.weight.data = procrustes(en_new, fr_new, new_enfr_map.weight.data, real_dico_s2t)
new_fren_map.weight.data = procrustes(fr_new, en_new, new_fren_map.weight.data, real_dico_t2s)

# Evaluate optimal mappings
old_s2t, old_t2s, _, _ = evaluate_uwt(old_enfr_map, old_fren_map, enfr, fren, en_old, fr_old)
new_s2t, new_t2s, _, _ = evaluate_uwt(new_enfr_map, new_fren_map, enfr, fren, en_new, fr_new)
print('Refined:')
print(old_s2t['csls']['score'], new_s2t['csls']['score'])
