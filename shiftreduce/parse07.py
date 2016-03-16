import itertools
import numpy
import sys
from argparse import ArgumentParser
from nltk.tree import Tree
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from util.constituent import *
from util.generators import batch
from util.functions import trace
from util.vocabulary import Vocabulary
from util.slstm import slstm

def parse_args():
  def_gpu_device = 0
  def_vocab = 32768
  def_queue = 256
  def_stack = 256
  def_epoch = 20
  def_minibatch = 100
  def_unary_limit = 3

  p = ArgumentParser(
    description='Shift-reduce constituent parser',
    usage=
      '\n  %(prog)s train [options] source model'
      '\n  %(prog)s test source model'
      '\n  %(prog)s -h',
  )

  p.add_argument('mode',
    help='\'train\' or \'test\'')
  p.add_argument('source',
    help='[in] source corpus'
      '\n  train: PENN-style constituent tree in each row'
      '\n  test: space-separated word list in each row')
  p.add_argument('model',
    help='[in/out] model parefix')
  p.add_argument('--use-gpu', action='store_true', default=False,
    help='use GPU calculation')
  p.add_argument('--gpu-device', default=def_gpu_device, metavar='INT', type=int,
    help='GPU device ID to be used (default: %(default)d)')
  p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
    help='vocabulary size (default: %d)' % def_vocab)
  p.add_argument('--queue', default=def_queue, metavar='INT', type=int,
    help='queue state size (default: %d)' % def_queue)
  p.add_argument('--stack', default=def_stack, metavar='INT', type=int,
    help='stack state size (default: %d)' % def_stack)
  p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
    help='number of training epoch (default: %d)' % def_epoch)
  p.add_argument('--minibatch', default=def_minibatch, metavar='INT', type=int,
    help='minibatch size (default: %d)' % def_minibatch)
  p.add_argument('--unary-limit', default=def_unary_limit, metavar='INT', type=int,
    help='maximum length of unary chain (default: %d)' % def_unary_limit)

  args = p.parse_args()

  # check args
  try:
    if args.mode not in ['train', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
    if args.vocab < 1: raise ValueError('you must set --vocab >= 1')
    if args.queue < 1: raise ValueError('you must set --queue >= 1')
    if args.stack < 1: raise ValueError('you must set --stack >= 1')
    if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
    if args.minibatch < 1: raise ValueError('you must set --minibatch >= 1')
  except Exception as ex:
    p.print_usage(file=sys.stderr)
    print(ex, file=sys.stderr)
    sys.exit()

  return args

class XP:
  __lib = None

  @staticmethod
  def set_library(args):
    if args.use_gpu:
      XP.__lib = cuda.cupy
      cuda.get_device(args.gpu_device).use()
    else:
      XP.__lib = numpy

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

class LinearEncoder(Chain):
  def __init__(self, n_input, n_output):
    super(LinearEncoder, self).__init__(
      w_xy = links.EmbedID(n_input, 4 * n_output),
      w_yy = links.Linear(n_output, 4 * n_output),
    )

  def __call__(self, c, x, y):
    return functions.lstm(c, self.w_xy(x) + self.w_yy(y))

class OperationEstimator(Chain):
  def __init__(self, n_queue, n_stack):
    super(OperationEstimator, self).__init__(
      w_qo = links.Linear(n_queue, NUM_OP),
      w_so = links.Linear(n_stack, NUM_OP),
    )

  def __call__(self, q, s):
    return functions.tanh(self.w_qo(q) + self.w_so(s))

class ShiftTransition(Chain):
  def __init__(self, n_queue, n_stack):
    super(ShiftTransition, self).__init__(
      w_qs = links.Linear(n_queue, n_stack),
      w_s1s = links.Linear(n_stack, n_stack),
    )

  def __call__(self, q, s1):
    return functions.tanh(self.w_qs(q) + self.w_s1s(s1))

class UnaryTransition(Chain):
  def __init__(self, n_queue, n_stack):
    super(UnaryTransition, self).__init__(
      w_qs = links.Linear(n_queue, 4 * n_stack),
      w_s1s = links.Linear(n_stack, 4 * n_stack),
      w_s2s = links.Linear(n_stack, 4 * n_stack),
    )

  def __call__(self, c, q, s1, s2):
    return functions.lstm(c, self.w_qs(q) + self.w_s1s(s1) + self.w_s2s(s2))

class BinaryTransition(Chain):
  def __init__(self, n_queue, n_stack):
    super(BinaryTransition, self).__init__(
      w_qs1 = links.Linear(n_queue, 4 * n_stack),
      w_s1s1 = links.Linear(n_stack, 4 * n_stack),
      w_s2s1 = links.Linear(n_stack, 4 * n_stack),
      w_s1s2 = links.Linear(n_stack, 4 * n_stack),
      w_s2s2 = links.Linear(n_stack, 4 * n_stack),
      w_s3s2 = links.Linear(n_stack, 4 * n_stack),
    )

  def __call__(self, c1, c2, q, s1, s2, s3):
    return slstm(
      c1, c2,
      self.w_qs1(q) + self.w_s1s1(s1) + self.w_s2s1(s2),
      self.w_s1s2(s1) + self.w_s2s2(s2) + self.w_s3s2(s3))

class LabelEstimator(Chain):
  def __init__(self, n_input, n_output):
    super(LabelEstimator, self).__init__(
        w_xy = links.Linear(n_input, n_output)
    )

  def __call__(self, x):
    return self.w_xy(x)

class Parser(Chain):
  def __init__(self, n_vocab, n_queue, n_stack, n_phrase, n_semiterminal):
    super(Parser, self).__init__(
      net_encoder = LinearEncoder(n_vocab, n_queue),
      net_operation = OperationEstimator(n_queue, n_stack),
      net_shift = ShiftTransition(n_queue, n_stack),
      net_unary = UnaryTransition(n_queue, n_stack),
      net_binary = BinaryTransition(n_queue, n_stack),
      net_phrase = LabelEstimator(n_stack, n_phrase),
      net_semiterminal = LabelEstimator(n_stack, n_semiterminal),
    )
    self.n_vocab = n_vocab
    self.n_queue = n_queue
    self.n_stack = n_stack
    self.n_phrase = n_phrase
    self.n_semiterminal = n_semiterminal

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.n_vocab, file=fp)
      print(self.n_queue, file=fp)
      print(self.n_stack, file=fp)
      print(self.n_phrase, file=fp)
      print(self.n_semiterminal, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      n_vocab = int(next(fp))
      n_queue = int(next(fp))
      n_stack = int(next(fp))
      n_phrase = int(next(fp))
      n_semiterminal = int(next(fp))
      return Parser(n_vocab, n_queue, n_stack, n_phrase, n_semiterminal)

  def forward(self, word_list, gold_op_list, unary_limit):
    is_training = gold_op_list is not None

    # check args
    if len(word_list) < 1:
      raise ValueError('Word list is empty.')
    if is_training:
      n_shift = 0
      n_binary = 0
      for op, _ in gold_op_list:
        if op == OP_SHIFT: n_shift += 1
        if op == OP_BINARY: n_binary += 1
      if n_shift != len(word_list) or n_binary != len(word_list) - 1:
        raise ValueError(
            'Invalid operation number: SHIFT=%d (required: %d), BINARY=%d (required: %d)' %
            (n_shift, n_binary, len(word_list), len(word_list) - 1))
      if gold_op_list[-1] != (OP_FINISH, None):
        raise ValueError('Last operation is not OP_FINISH.')

    # initial values
    OP_ZEROS = XP.fzeros((1, NUM_OP))
    QUEUE_ZEROS = XP.fzeros((1, self.n_queue))
    STACK_ZEROS = XP.fzeros((1, self.n_stack))
    NEG_INF = -1e20

    # queue encoding
    q_list = []
    qc = QUEUE_ZEROS
    q = QUEUE_ZEROS
    
    for text, wid in reversed(word_list):
      qc, q = self.net_encoder(qc, XP.iarray([wid]), q)
      q_list.insert(0, (text, q))

    # estimate
    s_list = []
    unary_chain = 0
    if is_training:
      loss = XP.fzeros(())

    for i in itertools.count():
      text, q = q_list[0] if q_list else ('', QUEUE_ZEROS)
      t1, sc1, s1 = s_list[-1] if s_list else (None, STACK_ZEROS, STACK_ZEROS)
      t2, sc2, s2 = s_list[-2] if len(s_list) >= 2 else (None, STACK_ZEROS, STACK_ZEROS)
      t3, sc3, s3 = s_list[-3] if len(s_list) >= 3 else (None, STACK_ZEROS, STACK_ZEROS)

      o = self.net_operation(q, s1)

      if is_training:
        loss += functions.softmax_cross_entropy(o, XP.iarray([gold_op_list[i][0]]))
        o_argmax = gold_op_list[i][0]
      else:
        o_filter = [0.0 for _ in range(NUM_OP)]
        filtered = 0
        if not q_list:
          o_filter[OP_SHIFT] = NEG_INF
          filtered += 1
        if not s_list or unary_chain >= unary_limit:
          o_filter[OP_UNARY] = NEG_INF
          filtered += 1
        if len(s_list) < 2:
          o_filter[OP_BINARY] = NEG_INF
          filtered += 1
        if q_list or len(s_list) > 1:
          o_filter[OP_FINISH] = NEG_INF
        if filtered == NUM_OP:
          raise RuntimeError('No possible operation!')

        o += XP.farray([o_filter])
        o_argmax = int(cuda.to_cpu(o.data.argmax(1)))

      if o_argmax == OP_SHIFT:
        t0 = Tree(None, [text])
        sc0, s0 = (STACK_ZEROS, self.net_shift(q, s1))
        q_list.pop(0)
        unary_chain = 0
        label = self.net_semiterminal(s0)
      elif o_argmax == OP_UNARY:
        t0 = Tree(None, [t1])
        sc0, s0 = self.net_unary(sc1, q, s1, s2)
        s_list.pop()
        unary_chain += 1
        label = self.net_phrase(s0)
      elif o_argmax == OP_BINARY:
        t0 = Tree(None, [t2, t1])
        sc0, s0 = self.net_binary(sc1, sc2, q, s1, s2, s3)
        s_list.pop()
        s_list.pop()
        unary_chain = 0
        label = self.net_phrase(s0)
      else: # OP_FINISH
        break

      if is_training:
        loss += functions.softmax_cross_entropy(label, XP.iarray([gold_op_list[i][1]]))
        label_argmax = gold_op_list[i][1]
      else:
        label_argmax = int(cuda.to_cpu(label.data.argmax(1)))

      t0.set_label(label_argmax)
      s_list.append((t0, sc0, s0))

      '''
      if is_training:
        o_est = int(cuda.to_cpu(o.data.argmax(1)))
        label_est = int(cuda.to_cpu(label.data.argmax(1)))
        trace('%c %c gold=%d-%2d, est=%d-%2d, stack=%2d, queue=%2d' % (
            '*' if o_est == gold_op_list[i][0] else ' ',
            '*' if label_est == gold_op_list[i][1] else ' ',
            gold_op_list[i][0], gold_op_list[i][1],
            o_est, label_est,
            len(s_list), len(q_list)))
      '''

    if is_training:
      return loss
    else:
      # combine multiple trees if they exists, and return the result.
      t0, _, __ = s_list.pop()
      if s_list:
        raise RuntimeError('There exist multiple subtrees!')
      return t0

def train(args):
  trace('loading corpus ...')
  with open(args.source) as fp:
    trees = [make_tree(l) for l in fp]

  trace('extracting leaf nodes ...')
  word_lists = [extract_words(t) for t in trees]
  lower_lists = [[w.lower() for w in words] for words in word_lists]

  trace('extracting gold operations ...')
  op_lists = [make_operations(t) for t in trees]

  trace('making vocabulary ...')
  word_vocab = Vocabulary.new(lower_lists, args.vocab)
  phrase_set = set()
  semiterminal_set = set()
  for tree in trees:
    phrase_set |= set(extract_phrase_labels(tree))
    semiterminal_set |= set(extract_semiterminals(tree))
  phrase_vocab = Vocabulary.new([list(phrase_set)], len(phrase_set), add_special_tokens=False)
  semiterminal_vocab = Vocabulary.new([list(semiterminal_set)], len(semiterminal_set), add_special_tokens=False)

  trace('converting data ...')
  word_lists = [convert_word_list(x, word_vocab) for x in word_lists]
  op_lists = [convert_op_list(x, phrase_vocab, semiterminal_vocab) for x in op_lists]

  trace('start training ...')
  parser = Parser(
    args.vocab, args.queue, args.stack,
    len(phrase_set), len(semiterminal_set),
  )
  if args.use_gpu:
    parser.to_gpu()
  opt = optimizers.AdaGrad(lr = 0.005)
  opt.setup(parser)
  opt.add_hook(optimizer.GradientClipping(5))

  for epoch in range(args.epoch):
    n = 0
    
    for samples in batch(zip(word_lists, op_lists), args.minibatch):
      parser.zerograds()
      loss = XP.fzeros(())

      for word_list, op_list in zip(*samples):
        trace('epoch %3d, sample %6d:' % (epoch + 1, n + 1))
        loss += parser.forward(word_list, op_list, 0)
        n += 1
      
      loss.backward()
      opt.update()

    trace('saving model ...')
    prefix = args.model + '.%03.d' % (epoch + 1)
    word_vocab.save(prefix + '.words')
    phrase_vocab.save(prefix + '.phrases')
    semiterminal_vocab.save(prefix + '.semiterminals')
    parser.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', parser)

  trace('finished.')

def test(args):
  trace('loading model ...')
  word_vocab = Vocabulary.load(args.model + '.words')
  phrase_vocab = Vocabulary.load(args.model + '.phrases')
  semiterminal_vocab = Vocabulary.load(args.model + '.semiterminals')
  parser = Parser.load_spec(args.model + '.spec')
  if args.use_gpu:
    parser.to_gpu()
  serializers.load_hdf5(args.model + '.weights', parser)

  trace('generating parse trees ...')
  with open(args.source) as fp:
    for l in fp:
      word_list = convert_word_list(l.split(), word_vocab)
      tree = combine_xbar(
          restore_labels(
              parser.forward(word_list, None, args.unary_limit),
              phrase_vocab,
              semiterminal_vocab))
      print('( ' + tree_to_string(tree) + ' )')

  trace('finished.')

def main():
  args = parse_args()
  XP.set_library(args)
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)

if __name__ == '__main__':
  main()

