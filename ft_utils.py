import io
import os
from math import pi

import torch
from torchtext import vocab

import fasttext
from fasttext.FastText import _FastText as fasttext_model
USE_CUDA= torch.cuda.is_available()

class fastTextWrapper():

    def __init__(self, fasttext_model, vectors, max_vocab=-1):
        # self.fasttext_model = fasttext_model
        self.om = torch.FloatTensor(fasttext_model.get_output_matrix())
        self.vocab = vectors
        self.vectors = vectors.vectors
        self.dim = vectors.dim

        assert isinstance(max_vocab, int) and (max_vocab > 0 or max_vocab == -1)
        self.max_vocab = max_vocab

        self.om = torch.FloatTensor(fasttext_model.get_output_matrix())[:max_vocab]

        words, word_freqs = fasttext_model.get_words(include_freq=True)
        self.words, self.word_freqs = words[:max_vocab], word_freqs[:max_vocab]

    def get_nns(self, word, k):
        v = self.get_vec(word)
        inds = self.knn(v, k)
        return [self.words[i] for i in inds]

    def knn(self, vec, k):
        cos = (vec / vec.norm()).matmul((self.vectors / self.vectors.norm(dim=1)).transpose(0,1))
        _, knn_inds = cos.topk(k)
        return knn_inds

    def get_vec(self, word):
        assert word in self.words
        ind = self.words.index(word)
        return self.vectors[ind]

    def get_out_dist(self, vec):
        energies = vec.matmul(self.om.transpose(0,1))
        return self.expit(energies)

    def kld(self, o1, o2, weights=None):
        if weights is None:
            XE = -((o1 * o2.log()).mean(dim=1) + ((1 - o1) * (1 - o2).log()).mean(dim=1))
            SE = -((o1 * o1.log()).mean(dim=1) + ((1 - o1) * (1 - o1).log()).mean(dim=1))
        else:
            XE = (-((o1 * o2.log()) + ((1 - o1) * (1 - o2).log())) * weights).sum(dim=1)
            SE = (-((o1 * o1.log()) + ((1 - o1) * (1 - o1).log())) * weights).sum(dim=1)
        return XE - SE

    def jsd(self, o1, o2, weights=None):
        o12 = 0.5 * (o1 + o2)
        return 0.5 * (self.kld(o1, o12, weights=weights) +
                      self.kld(o2, o12, weights=weights))

    def expit(self, x):
        return 1 / (1 + (-x).exp())

def get_fasttext_model(model_language, max_vocab=100000, model_dir='fasttext'):
    assert isinstance(model_language, str)
    model_name = 'wiki.' + model_language + '.bin'
    vectors_name = 'wiki.' + model_language + '.vec'
    if model_dir is None:
        model_dir = os.getcwd()

    model_path = os.path.join(model_dir, model_name)
    vectors_path = os.path.join(model_dir, vectors_name)

    assert os.path.exists(model_path)
    assert os.path.exists(vectors_path)
    assert isinstance(max_vocab, int) and (max_vocab > 0 or max_vocab == -1)
    ftm = fasttext_model(model_path)
    vec = vocab.Vectors(vectors_path, max_vectors=max_vocab)

    return fastTextWrapper(ftm , vec, max_vocab)

def cosine_sim(x, y):
    import pdb; pdb.set_trace()
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    assert x.dim() == y.dim() == 2 and x.size(-1) == y.size(-1)
    cos = (x / x.norm(dim=1)).matmul((y / y.norm(dim=1).unsqueeze(1)).transpose(0,1))
    return cos

def angular_dist(x, y):
    cos = cosine_sim(x, y)
    return torch.acos(cos) / pi


def load_muse_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """

    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    print("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def procrustes(src_emb, tgt_emb, mapping, pairs):
    """
    Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    if pairs.size()[0] == 0:
        print('No dictionary for procrustes')
        return mapping
    else:
        A = src_emb[pairs[:, 0]]
        B = tgt_emb[pairs[:, 1]]
        W = mapping
        M = B.transpose(0, 1).mm(A)
        U, S, V = torch.svd(M)
        #scipy.linalg.svd(M, full_matrices=True)

        return U.mm(V.t()).type_as(W)

def evaluate_uwt(s2t_mapping, t2s_mapping, s2t_pairs, t2s_pairs, src_vecs, tgt_vecs,
                      modes=['cosine', 'csls'], N=-1):

    N_s2t = s2t_pairs.size(0) if N <= 0 or N >= s2t_pairs.size(0) else N
    N_t2s = t2s_pairs.size(0) if N <= 0 or N >= t2s_pairs.size(0) else N

    '''
    with torch.no_grad():
        if 'csls' in modes:
            r_Y_tgt = build_graph(tgt_vocab)
            r_Y_src = build_graph(src_vocab)
    '''

    top1_s2t, top5_s2t, skipped_s2t = _eval(s2t_pairs, src_vecs, tgt_vecs, s2t_mapping, modes, N=N_s2t)
    top1_t2s, top5_t2s, skipped_t2s = _eval(t2s_pairs, tgt_vecs, src_vecs, t2s_mapping, modes, N=N_t2s)

    #print('Total skipped (S2T): {}'.format(skipped_s2t))

    return top1_s2t, top1_t2s, top5_s2t, top5_t2s


def _eval(pairs, src_vecs, tgt_vecs, s2t_mapping, modes, N):
    uniques_src, counts_src = pairs[:N, 0].unique(return_counts=True, sorted=True)
    dim = src_vecs.size(1)
    M = uniques_src.size(0)
    top1 = {mode: {"indices":torch.zeros(M)} for mode in modes}
    top5 = {mode: {"indices":torch.zeros(M)} for mode in modes}
    count=0
    skipped=0

    unique_num, unique_denom, many_num, many_denom = 0, 0, 0, 0

    with torch.no_grad():
        # 1. Convert all vectors to other spoce
        assert (pairs[:N, 0] >= src_vecs.size(0)).sum() == 0

        emb1 = s2t_mapping(src_vecs)[:, :dim]
        emb2 = tgt_vecs.clone()
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        query = emb1[pairs[:N, 0]]

        # 2. Find NNs of each under different modes
        NNs = {}
        for mode in modes:
            if mode == 'cosine':
                nn_inds = cosine_knn(query, emb2, k=5)
            elif mode == 'csls':
                nn_inds = CSLS(query, emb1, emb2, k=5)
            NNs[mode] = nn_inds

        # 3. Calculate metric

        for i in range(M):
            if uniques_src[i].item() > src_vecs.size(0):
                skipped += 1
                continue

            count_i = counts_src[i]
            for mode in modes:
                for j in range(count_i):
                    outind = pairs[count + j, 1]

                    if outind in NNs[mode][count + j, :1]:
                        top1[mode]["indices"][i] = 1
                    if outind in NNs[mode][count + j]:
                        top5[mode]["indices"][i] = 1

            count += count_i
            if count_i > 1:
                many_denom += 1
                many_num += top1['cosine']["indices"][i]
            else:
                unique_denom += 1
                unique_num += top1['cosine']["indices"][i]
        '''
        for i in range(M):
            if pairs[i,0].item() > src_vecs.size(0):
                skipped += 1
                continue
            src_vec = src_vecs[uniques_src[i], :]
            tgt_vec = s2t_mapping(src_vec)
            count_i = counts_src[i]
            for mode in modes:
                topk = get_nns(tgt_vec, tgt_vocab, 5, mode)
                for j in range(count_i):
                    if pairs[count+j,1] in topk[:1]:
                        top1[mode]["indices"][i] = 1
                    if pairs[count+j,1] in topk:
                        top5[mode]["indices"][i] = 1
                #top5[mode]["indices"][i] /= min(count_i, 5)
        '''



    for mode in modes:
        top1[mode]["score"] = top1[mode]["indices"].mean()
        top5[mode]["score"] = top5[mode]["indices"].mean()

    # print('Unique frac: {}\nMany frac: {}'.format((unique_num / unique_denom), (many_num / many_denom)))

    return top1, top5, skipped


def get_nns(mapping, query, vectors, k, mode='cosine'):
    assert mode in ['euclidean', 'cosine']

    query_mapped = mapping(query)[:, :300]

    if mode == 'euclidean':
        return knn(query_mapped, vectors, k)

    elif mode == 'cosine':
        query_mapped /= query_mapped.norm(dim=1, keepdim=True)
        return cosine_knn(query_mapped, vectors, k)

    elif mode == 'csls':
        query_mapped /= query_mapped.norm(dim=1, keepdim=True)
        return CSLS(query_mapped, vectors, k)


def knn(candidate, vectors, k=5, vocab=None):
    if USE_CUDA:
        candidate = candidate.cuda()
        vectors = vectors.cuda()

    temp = candidate.expand_as(vectors)
    dists = torch.sum((temp - vectors) ** 2, 1)

    '''
        if vocab is not None:
            for i in range(k):
                print('Nearest Neighbour {}: {}'.format(i, vocab.itos[knn_out[i]]))
            print('\n')
        '''
    return (-1 * dists).topk(k)


def cosine_knn(query, vectors, k=5, vocab=None, bs=1000):
    if query.size(0) > bs:
        ret = torch.zeros((query.size(0), k)).long()
        if USE_CUDA:
            ret = ret.cuda()
        for i in range(0, int(query.size(0)), bs):
            ret[i:min(i + bs, query.size(0))] = cosine_knn(query[i:min(i + bs, query.size(0))], vectors, k=k, vocab=vocab, bs=bs)
            #scores[i * bs:max((i+1) * bs, query.size(0))] = query[i * bs:max((i+1) * bs, query.size(0))].mm(vectors.transpose(0, 1))
        return ret
    else:
        scores = query.mm(vectors.transpose(0, 1))
    return scores.topk(k, 1, True)[1]

def frac_knn(candidate, vectors, k=5, vocab=None, p=0.5):
    if USE_CUDA:
        candidate = candidate.cuda()
        vectors = vectors.cuda()
    temp = candidate.expand_as(vectors)
    dists = torch.sum((temp - vectors) ** p, 1)
    '''
        if vocab is not None:
            for i in range(k):
                print('Nearest Neighbour {}: {}'.format(i, vocab.itos[knn_out[i]]))
            print('\n')
        '''
    return (-1 * dists).topk(k)

def scaled_knn(candidate, vectors, scale=None, k=5, vocab=None):
    if scale is None:
        return knn(candidate, vectors, k, vocab)
    else:
        if USE_CUDA:
            candidate = candidate.cuda()
            vectors = vectors.cuda()
            scale = scale.cuda()
        temp = candidate.expand_as(vectors)
        dists = torch.sum(((temp - vectors)/ scale) ** 2, 1)
        '''
            if vocab is not None:
                for i in range(k):
                    print('Nearest Neighbour {}: {}'.format(i, vocab.itos[knn_out[i]]))
                print('\n')
            '''
        return (-1 * dists).topk(k)

def CSLS(query, emb1, emb2, k=5):
    '''
    query = word mapped into tgt vector space
    emb1 = all src vectors mapped into tgt space
    emb2 = all tgt vectors
    '''
    with torch.no_grad():
        average_dist1 = get_nn_avg_dist(emb2, query, 10)
        average_dist2 = get_nn_avg_dist(emb1, emb2, 10)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)

        bs = 128
        knn_inds = torch.zeros(query.size(0), k).long()
        knn_inds = knn_inds.cuda() if USE_CUDA else knn_inds
        temp_emb = emb2.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            temp_scores = query[i:i + bs].mm(temp_emb)
            temp_scores.mul_(2)
            temp_scores.sub_(average_dist1[:, None][i:i + bs])
            temp_scores.sub_(average_dist2[None, :])
            knn_inds[i:i + bs] = temp_scores.topk(k, 1, True)[1]
        #scores = query.mm(emb2.transpose(0, 1))

    del(average_dist1)
    del(average_dist2)
    return knn_inds


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    """
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    del(emb)
    return all_distances.numpy()

def mean_cos(query, emb1, emb2):
    NNs = CSLS(query, emb1, emb2, k=1).squeeze(1)
    with torch.no_grad():
        return (query * emb2[NNs]).sum(1).mean()

