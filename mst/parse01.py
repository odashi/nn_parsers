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
  def_epoch = 20

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
  __train = False

  @staticmethod
  def set_library(args):
    if args.gpu >= 0:
      XP.__lib = cuda.cupy
      cuda.get_device(args.gpu).use()
    else:
      XP.__lib = numpy
    XP.__train = args.mode == 'train'

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
    return functions.dropout(x, ratio=0.5, train=XP.__train)

def conll_to_train(conll, vocab):
  return [(vocab.stoi(x[1]), x[6]) for x in conll]

class Embed(Chain):
  def __init__(self, vocab_size, embed_size):
    super(Embed, self).__init__(
      x_e = links.EmbedID(vocab_size, embed_size),
    )

  def __call__(self, x):
    return functions.tanh(self.x_e(x))

class StackedLSTM(ChainList):
  def __init__(self, input_size, output_size):
    super(StackedLSTM, self).__init__(
      links.LSTM(input_size, output_size),
      links.LSTM(output_size, output_size),
      #links.LSTM(output_size, output_size),
    )

  def reset_state(self):
    self[0].reset_state()
    self[1].reset_state()
    #self[2].reset_state()

  def __call__(self, x):
    h = self[0](x)
    h = self[1](h)
    #h = self[2](h)
    return h

class Combiner(Chain):
  def __init__(self, input_size, output_size):
    super(Combiner, self).__init__(
      x1_y = links.Linear(input_size, output_size),
      x2_y = links.Linear(input_size, output_size),
    )

  def __call__(self, x1, x2):
    return functions.tanh(self.x1_y(x1) + self.x2_y(x2))

class Parser(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Parser, self).__init__(
      p_embed = Embed(vocab_size, embed_size),
      c_embed = Embed(vocab_size, embed_size),
      r_embed = Embed(vocab_size, embed_size),
      p_forward = StackedLSTM(embed_size, hidden_size),
      c_forward = StackedLSTM(embed_size, hidden_size),
      r_forward = StackedLSTM(embed_size, hidden_size),
      p_backward = StackedLSTM(embed_size, hidden_size),
      c_backward = StackedLSTM(embed_size, hidden_size),
      r_backward = StackedLSTM(embed_size, hidden_size),
      p_combine = Combiner(hidden_size, hidden_size),
      c_combine = Combiner(hidden_size, hidden_size),
      r_combine = Combiner(hidden_size, hidden_size),
      r_scorer = links.Linear(hidden_size, 1),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
  
  def reset_state(self):
    self.p_forward.reset_state()
    self.c_forward.reset_state()
    self.r_forward.reset_state()
    self.p_backward.reset_state()
    self.c_backward.reset_state()
    self.r_backward.reset_state()

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.vocab_size, file=fp)
      print(self.embed_size, file=fp)
      print(self.hidden_size, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      vocab_size = int(next(fp))
      embed_size = int(next(fp))
      hidden_size = int(next(fp))
      return Parser(vocab_size, embed_size, hidden_size)
  
  def forward(self, data):
    self.reset_state()
    
    x_list = [XP.iarray([d[0]]) for d in data]
    pe_list = [self.p_embed(x) for x in x_list]
    ce_list = [self.c_embed(x) for x in x_list]
    re_list = [self.r_embed(x) for x in x_list]

    pf_list = []
    for pe in pe_list:
      pf_list.append(self.p_forward(pe))

    cf_list = []
    for ce in ce_list:
      cf_list.append(self.c_forward(ce))

    rf_list = []
    for re in re_list:
      rf_list.append(self.r_forward(re))

    pb_list = []
    for pe in reversed(pe_list):
      pb_list.append(self.p_backward(pe))

    cb_list = []
    for ce in reversed(ce_list):
      cb_list.append(self.c_backward(ce))

    rb_list = []
    for re in reversed(re_list):
      rb_list.append(self.r_backward(re))

    pc_list = [self.p_combine(pf, pb) for pf, pb in zip(pf_list, pb_list)]
    cc_list = [self.c_combine(cf, cb) for cf, cb in zip(cf_list, cb_list)]
    rc_list = [self.r_combine(rf, rb) for rf, rb in zip(rf_list, rb_list)]

    P = functions.reshape(
      functions.concat(pc_list, 0),
      (1, len(data), self.hidden_size))
    C = functions.reshape(
      functions.concat(cc_list, 0),
      (1, len(data), self.hidden_size))
    R = functions.concat(rc_list, 0)

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

  parser = Parser(args.vocab, args.embed, args.hidden)
  if args.gpu >= 0:
    parser.to_gpu()

  opt = optimizers.AdaGrad(lr=0.01)
  opt.setup(parser)
  opt.add_hook(optimizer.GradientClipping(10))
  opt.add_hook(optimizer.WeightDecay(0.0001))

  for epoch in range(args.epoch):
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
    parser.save_spec(prefix + '.parent_spec')
    serializers.save_hdf5(prefix + '.parent_weights', parser)

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

