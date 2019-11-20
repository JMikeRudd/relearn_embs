import torch
from ft_utils import *

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enmod = torch.load('trained_models/en_discrete/isometric_embedding').to(device)
frmod = torch.load('trained_models/fr_discrete/isometric_embedding').to(device)

# Get old and new vectors
en_old, fr_old = enmod.metric.ft_model.vectors.to(device), frmod.metric.ft_model.vectors.to(device)
en_new, fr_new = enmod.embed(torch.arange(200000).to(device)), frmod.embed(torch.arange(200000).to(device))

# Normalize old vectors
en_old /= en_old.norm(dim=1, keepdim=True)
en_old -= en_old.mean(0, keepdim=True)
en_old /= en_old.norm(dim=1, keepdim=True)
fr_old /= fr_old.norm(dim=1, keepdim=True)
fr_old -= fr_old.mean(0, keepdim=True)
fr_old /= fr_old.norm(dim=1, keepdim=True)

# Get Dictionaries
enfr = load_muse_dictionary('fasttext/en-fr.5000-6500.txt', enmod.metric.ft_model.vocab.stoi, frmod.metric.ft_model.vocab.stoi).to(device)
fren = load_muse_dictionary('fasttext/fr-en.5000-6500.txt', frmod.metric.ft_model.vocab.stoi, enmod.metric.ft_model.vocab.stoi).to(device)

# Get optimal mappings
old_dim, new_dim = enmod.metric.ft_model.dim, enmod.emb_space.emb_dim
old_enfr_map = torch.nn.Linear(old_dim, old_dim, bias=False).to(device)
old_fren_map = torch.nn.Linear(old_dim, old_dim, bias=False).to(device)
new_enfr_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)
new_fren_map = torch.nn.Linear(new_dim, new_dim, bias=False).to(device)


old_enfr_map.weight.data = procrustes(en_old, fr_old, old_enfr_map.weight.data, enfr)
old_fren_map.weight.data = procrustes(fr_old, en_old, old_fren_map.weight.data, fren)
new_enfr_map.weight.data = procrustes(en_new, fr_new, new_enfr_map.weight.data, enfr)
new_fren_map.weight.data = procrustes(fr_new, en_new, new_fren_map.weight.data, fren)

# Evaluate optimal mappings
old_s2t, old_t2s, _, _ = evaluate_uwt(old_enfr_map, old_fren_map, enfr, fren, en_old, fr_old)
new_s2t, new_t2s, _, _ = evaluate_uwt(new_enfr_map, new_fren_map, enfr, fren, en_new, fr_new)
