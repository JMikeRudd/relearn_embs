import argparse
import logging
import os
import copy
import wget

from ft_utils import get_fasttext_model
from models.isometric_embedding import *
from models.embedding_space import *
from models.embedding_models import *
from models.metrics import *
from models.utils   import *


from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset


def main(lang, vec_dir='fasttext/', max_vocab=200000,
         emb_space_type='spherical', emb_model_type='Discrete', metric_type='ftJSD', n_class=50000,
         proportional=False, emb_dim=300, hidden_size=None, n_layers=1,
         lr=0.001, opt_cls='Adam', epochs=20, batch_size=128,
         model_dir=None, name=None, save_every=5, print_every=1, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    if model_dir is None:
        model_dir = os.path.join(os.getcwd(), 'trained_models')

    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    assert name is not None and isinstance(name, str)

    save_dir = os.path.join(model_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all arguments of the current training run for inspection later if desired
    logging.basicConfig(filename=os.path.join(save_dir, 're_embed_train.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    train_args = copy(locals())
    logger.info('Train Arguments:')
    kmaxlen = max([len(k) for k in train_args.keys()])
    formatStr = '\t{:<' + str(kmaxlen) + '} {}'
    for k in sorted(train_args.keys()):
        logger.info(formatStr.format(k, train_args[k]))

    # Default embedding paths from language names
    lang_path = os.path.join(vec_dir, 'wiki.{}.vec'.format(lang))
    # Download embeddings if not present
    if not os.path.exists(lang_path):
        vec_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(lang)
        try:
            _ = wget.download(vec_url, out=vec_dir)
        except:
            print('Download of {} vectors from {} failed.'.format(lang, vec_url))

    # vocab = vocab.Vectors(lang_path, max_vectors=max_vocab)
    ft_mod = get_fasttext_model(model_language=lang, max_vocab=max_vocab)
    ft_mod.vectors = ft_mod.vectors.to(device)
    ft_mod.om = ft_mod.om.to(device)

    # Declare model(s)
    # Get embedding model
    if emb_model_type == 'MLP':
        layers = [ft_mod.dim] + [hidden_size] * n_layers + [emb_dim]
        emb_model = MLPEmbMapping(layers)

        metric = fastTextJSDMetric(ft_mod, n_class=n_class)

        loader = DataLoader(TensorDataset(ft_mod.vectors.to(device)),
                        shuffle=True, batch_size=batch_size)
    elif emb_model_type == 'Discrete':
        emb_model = DiscreteEmbMapping(max_vocab, emb_dim)

        metric = DiscretefastTextJSDMetric(ft_mod, n_class=n_class)

        loader = DataLoader(TensorDataset(torch.arange(max_vocab).to(device)),
                            shuffle=True, batch_size=batch_size)

    # Get embedding space
    emb_space = get_embedding_space(emb_space_type, emb_model)

    '''
    # Get Metric
    if metric_type == 'ftJSD':
        
    else:
        raise ValueError('Just use ftJSD')
    '''

    # Declare Isometric Embedding
    isom_embedding = IsometricEmbedding(emb_space, metric,
                                        proportional=proportional).to(device)

    # Declare optim(s)
    if opt_cls == 'Adam':
        opt = Adam(isom_embedding.parameters(), lr=lr)
    elif opt_cls == 'SGD':
        opt = SGD(isom_embedding.parameters(), lr=lr)
    else:
        raise ValueError('{} not a supported opt_cls'.format(opt_cls))

    # Train model
    train_isometric_embedding(isom_embedding, epochs, loader, opt,
                              print_every=print_every, save_every=save_every,
                              save_dir=save_dir)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data Arguments
    parser.add_argument('--vec_dir', type=str, default='fasttext/')
    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--max_vocab', help='how many vectors to translate to',
                        type=int, default=200000)

    # Embedding Space Arguments
    parser.add_argument('--emb_space_type',
                        help='what type of embedding space to use',
                        type=str, default=None,
                        choices=EMBEDDING_SPACES)
    parser.add_argument('--metric_type',
                        help='what type of metric to use on real data',
                        type=str, default=None,
                        choices=METRICS)
    parser.add_argument('--n_class', help='how many classes to consider',
                        type=int, default=50000)
    parser.add_argument('--emb_model_type',
                        help='what type of emb model to use',
                        type=str, default='Dicrete',
                        choices=['MLP', 'Discrete'])
    parser.add_argument('--proportional', dest='proportional',
                        default=False, action='store_const', const=True)

    # Model Specifications
    parser.add_argument('--emb_dim', help='size of embeddings',
                        type=int, default=300)
    parser.add_argument('--hidden_size', help='dimension of hidden layers',
                        type=int, default=2048)
    parser.add_argument('--n_layers', help='number of layers for embedding model',
                        type=int, default=1)
    
    # Optimization Arguments
    parser.add_argument('--lr', help='outer learning rate', type=float, default=0.001)
    parser.add_argument('--opt_cls', help='which optimizer type to use',
                        type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--epochs', help='epochs', type=int, default=250)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)

    # Recording Args
    parser.add_argument('--model_dir', help='directory to save models in', type=str, default=None)
    parser.add_argument('--name', help='what to name model', type=str, default=None)
    parser.add_argument('--save_every', help='how often to save (epochs)', type=int, default=5)
    parser.add_argument('--print_every', help='how often to print (epochs)', type=int, default=1)
    parser.add_argument('--seed', help='random seed', type=str, default=None)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    main(**args)
