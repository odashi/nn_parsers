import numpy
import sys
import random
from argparse import ArgumentParser
from chainer import Chain, ChainList, Variable, cuda, functions, links, optimizer, optimizers, serializers
from collections import defaultdict
from util.functions import trace

def parse_args():
  def_gpu = -1
  def_vocab = 32768
  def_embed = 1024
  def_hidden = 256
  def_depth = 3
  def_epoch = 50

  p = ArgumentParser(
    description='Neural MST dependency parser',
    usage=
      '\n %(prog)s --mode=train --train=FILE --dev=FILE --model=FILE [options]'
      '\n %(prog)s --mode=test --model=FILE [options]'
      '\n %(prog)s -h'
  )

  p.add_argument('--mode',
    default=None, metavar='STR', type=str,
    help='\'train\' or \'test\'')
  p.add_argument('--train',
    default=None, metavar='STR', type=str,
    help='[in] training corpus with CoNLL dependency')
  p.add_argument('--dev',
    default=None, metavar='STR', type=str,
    help='[in] development corpus with CoNLL dependency')
  p.add_argument('--model',
    default=None, metavar='STR', type=str,
    help='[in/out] model file prefix')
  p.add_argument('--gpu',
    default=def_gpu, metavar='INT', type=int,
    help='GPU ID if use a GPU, negative value if use CPU (default: %(default)d)')
  p.add_argument('--vocab',
    default=def_vocab, metavar='INT', type=int,
    help='vocabulary size (default: %(default)d)')
  p.add_argument('--embed',
    default=def_embed, metavar='INT', type=int,
    help='embedding size (default: %(default)d)')
  p.add_argument('--hidden',
    default=def_hidden, metavar='INT', type=int,
    help='hidden size (default: %(default)d)')
  p.add_argument('--depth',
    default=def_depth, metavar='INT', type=int,
    help='encoder depth (default: %(default)d)')
  p.add_argument('--epoch',
    default=def_epoch, metavar='INT', type=int,
    help='maximum training epoch (default: %(default)d)')

  args = p.parse_args()

  # check args
  try:
    if args.mode == 'train':
      if not args.train: raise ValueError('you msut give a string for --train')
      if not args.dev: raise ValueError('you msut give a string for --dev')
      if not args.model: raise ValueError('you msut give a string for --model')
      if not args.vocab > 0: raise ValueError('you must set --vocab > 0')
      if not args.embed > 0: raise ValueError('you must set --embed > 0')
      if not args.hidden > 0: raise ValueError('you must set --hidden > 0')
      if not args.depth > 0: raise ValueError('you must set --depth > 0')
      if not args.epoch > 0: raise ValueError('you must set --epoch > 0')
    elif args.mode == 'test':
      if not args.model: raise ValueError('you msut give a string for --model')
    else:
      raise ValueError('you must set mode = \'train\' or \'test\'')
  except Exception as ex:
    p.print_usage(file=sys.stderr)
    print(ex, file=sys.stderr)
    sys.exit()

  return args

def read_conll(filename):
  data = []
  with open(filename) as fp:
    for l in fp:
      l = l.split()
      if not l:
        yield data
        data = []
      elif len(l) == 10:
        l[0] = int(l[0]) - 1
        l[6] = int(l[6]) - 1
        data.append(tuple(l))
      else:
        raise RuntimeError('invalid CoNLL format')
  if data:
    yield data

class Vocabulary:
  def __init__(self):
    pass

  def __len__(self):
    return self.__size

  def stoi(self, s):
    return self.__stoi[s]

  def itos(self, i):
    return self.__itos[i]

  @staticmethod
  def from_conll(filename, size):
    self = Vocabulary()
    self.__size = size

    freq = defaultdict(lambda: 0)
    for conll in read_conll(filename):
      for word in conll:
        freq[word[1]] += 1

    self.__stoi = defaultdict(lambda: 0)
    self.__stoi['<unk>'] = 0
    self.__itos = [''] * self.__size
    self.__itos[0] = '<unk>'
    
    for i, (k, _) in zip(range(1, self.__size), sorted(freq.items(), key=lambda x: -x[1])):
      self.__stoi[k] = i
      self.__itos[i] = k

    return self

  def save(self, filename):
    with open(filename, 'w') as fp:
      print(self.__size, file=fp)
      for i in range(self.__size):
        print(self.__itos[i], file=fp)

  @staticmethod
  def load(filename):
    with open(filename) as fp:
      self = Vocabulary()
      self.__size = int(next(fp))
      self.__stoi = defaultdict(lambda: 0)
      self.__itos = [''] * self.__size
      for i in range(self.__size):
        s = next(fp).strip()
        if s:
          self.__stoi[s] = i
          self.__itos[i] = s
    
    return self

class XP:
  __lib = None
  train = False

  @staticmethod
  def set_library(args):
    if args.gpu >= 0:
      XP.__lib = cuda.cupy
      cuda.get_device(args.gpu).use()
    else:
      XP.__lib = numpy
    XP.train = args.mode == 'train'

  @staticmethod
  def __zeros(shape, dtype):
    return Variable(XP.__lib.zeros(shape, dtype=dtype))

  @staticmethod
  def fzeros(shape):
    return XP.__zeros(shape, XP.__lib.float32)

  @staticmethod
  def __array(array, dtype):
    return Variable(XP.__lib.array(array, dtype=dtype))

  @staticmethod
  def iarray(array):
    return XP.__array(array, XP.__lib.int32)

  @staticmethod
  def farray(array):
    return XP.__array(array, XP.__lib.float32)

  @staticmethod
  def dropout(x):
    return functions.dropout(x, ratio=0.5, train=XP.train)

def conll_to_train(conll, vocab):
  return [(vocab.stoi(x[1]), x[6]) for x in conll]

class Embed(Chain):
  def __init__(self, vocab_size, embed_size):
    super(Embed, self).__init__(
      x_e = links.EmbedID(vocab_size, embed_size),
    )

  def __call__(self, x):
    return functions.tanh(self.x_e(x))

class Encoder(Chain):
  def __init__(self, input_size, output_size):
    super(Encoder, self).__init__(
      x_f = links.LSTM(input_size, output_size),
      x_b = links.LSTM(input_size, output_size),
      f_y = links.Linear(output_size, output_size),
      b_y = links.Linear(output_size, output_size),
    )

  def reset_state(self):
    self.x_f.reset_state()
    self.x_b.reset_state()

  def __call__(self, x_list):
    x_list = [XP.dropout(x) for x in x_list]
    f_list = []
    for x in x_list:
      f_list.append(XP.dropout(self.x_f(x)))
    b_list = []
    for x in reversed(x_list):
      b_list.insert(0, XP.dropout(self.x_b(x)))
    y_list = []
    for f, b in zip(f_list, b_list):
      y_list.append(functions.tanh(self.f_y(f) + self.b_y(b)))
    return y_list

class StackedEncoder(ChainList):
  def __init__(self, input_size, output_size, depth):
    encoders = [Encoder(input_size, output_size)]
    for _ in range(1, depth):
      encoders.append(Encoder(output_size, output_size))
    super(StackedEncoder, self).__init__(*encoders)

  def reset_state(self):
    for encoder in self:
      encoder.reset_state()

  def __call__(self, x_list):
    for encoder in self:
      x_list = encoder(x_list)
    return x_list

class Parser(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size, depth):
    super(Parser, self).__init__(
      p_embed = Embed(vocab_size, embed_size),
      c_embed = Embed(vocab_size, embed_size),
      r_embed = Embed(vocab_size, embed_size),
      p_encode = StackedEncoder(embed_size, hidden_size, depth),
      c_encode = StackedEncoder(embed_size, hidden_size, depth),
      r_encode = StackedEncoder(embed_size, hidden_size, depth),
      r_scorer = links.Linear(hidden_size, 1),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.depth = depth
  
  def reset_state(self):
    self.p_encode.reset_state()
    self.c_encode.reset_state()
    self.r_encode.reset_state()

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.vocab_size, file=fp)
      print(self.embed_size, file=fp)
      print(self.hidden_size, file=fp)
      print(self.depth, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      vocab_size = int(next(fp))
      embed_size = int(next(fp))
      hidden_size = int(next(fp))
      depth = int(next(fp))
      return Parser(vocab_size, embed_size, hidden_size, depth)
  
  def forward(self, data):
    self.reset_state()
    
    x_list = [XP.iarray([d[0]]) for d in data]
    ep_list = [self.p_embed(x) for x in x_list]
    ec_list = [self.c_embed(x) for x in x_list]
    er_list = [self.r_embed(x) for x in x_list]
    p_list = self.p_encode(ep_list)
    c_list = self.c_encode(ec_list)
    r_list = self.r_encode(er_list)

    P = functions.reshape(
      functions.concat(p_list, 0),
      (1, len(data), self.hidden_size))
    C = functions.reshape(
      functions.concat(c_list, 0),
      (1, len(data), self.hidden_size))
    R = functions.concat(r_list, 0)

    parent_scores = functions.reshape(
      functions.batch_matmul(C, P, transb=True),
      (len(data), len(data)))
    root_scores = functions.reshape(
      self.r_scorer(R),
      (1, len(data)))

    return parent_scores, root_scores
        
def train(args):
  vocab = Vocabulary.from_conll(args.train, args.vocab)
  train_dataset = [conll_to_train(x, vocab) for x in read_conll(args.train)]
  dev_dataset = [conll_to_train(x, vocab) for x in read_conll(args.dev)]

  parser = Parser(args.vocab, args.embed, args.hidden, args.depth)
  if args.gpu >= 0:
    parser.to_gpu()

  opt = optimizers.SGD(lr=0.1)
  opt.setup(parser)
  opt.add_hook(optimizer.GradientClipping(10))
  opt.add_hook(optimizer.WeightDecay(0.0001))

  for epoch in range(args.epoch):
    XP.train = True
    random.shuffle(train_dataset)
    parser.zerograds()
    loss = XP.fzeros(())
    
    for i, data in enumerate(train_dataset):
      trace('epoch %3d: train sample %6d:' % (epoch + 1, i + 1))
      parent_scores, root_scores = parser.forward(data)
      if len(data) > 1:
        parent_scores = functions.split_axis(parent_scores, len(data), 0)
      else:
        parent_scores = (parent_scores,)

      root = -1
      for j, (p_scores, (wid, parent)) in enumerate(zip(parent_scores, data)):
        if parent == -1:
          trace('  %3d: root' % j)
          root = j
        else:
          parent_est = p_scores.data.argmax()
          trace('%c %3d -> %3d (%3d)' % ('*' if parent == parent_est else ' ', j, parent_est, parent))
          loss += functions.softmax_cross_entropy(p_scores, XP.iarray([parent]))
      
      root_est = root_scores.data.argmax()
      trace('ROOT: %3d (%3d)' % (root_est, root))
      loss += functions.softmax_cross_entropy(root_scores, XP.iarray([root]))

      if (i+1) % 200 == 0:
        loss.backward()
        opt.update()
        parser.zerograds()
        loss = XP.fzeros(())

    loss.backward()
    opt.update()
    trace('epoch %3d: trained.                        ' % (epoch + 1))
    
    XP.train = False
    parent_num = 0
    parent_match = 0
    root_num = 0
    root_match = 0
    for i, data in enumerate(dev_dataset):
      trace('epoch %3d: dev sample %6d:' % (epoch + 1, i + 1), rollback=True)
      parent_scores, root_scores = parser.forward(data)
      if len(data) > 1:
        parent_scores = functions.split_axis(parent_scores, len(data), 0)
      else:
        parent_scores = (parent_scores,)

      root = -1
      for j, (p_scores, (wid, parent)) in enumerate(zip(parent_scores, data)):
        if parent == -1:
          root = j
        else:
          parent_est = p_scores.data.argmax()
          parent_num += 1
          parent_match += 1 if parent_est == parent else 0

      root_est = root_scores.data.argmax()
      root_num += 1
      root_match += 1 if root_est == root else 0

    result_str = \
      'epoch %3d: dev: parent-acc = %.4f (%5d/%5d), root-acc = %.4f (%4d/%4d)' % \
      ( \
        epoch + 1, \
        parent_match / parent_num, parent_match, parent_num, \
        root_match / root_num, root_match, root_num)
    trace(result_str)
    
    with open(args.model + '.log', 'a') as fp:
      print(result_str, file=fp)

    trace('epoch %3d: saving models ...' % (epoch + 1))
    prefix = args.model + '.%03d' % (epoch + 1)
    vocab.save(prefix + '.vocab')
    parser.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', parser)

    opt.lr *= 0.9

  trace('finished.')

def test(args):
  pass

def main():
  args = parse_args()
  XP.set_library(args)
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)

if __name__ == '__main__':
  main()

