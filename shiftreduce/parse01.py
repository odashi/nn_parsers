# Switch to toggle CPU/GPU operation
USE_GPU = False

import itertools
import sys
from argparse import ArgumentParser
from nltk.tree import Tree
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from util.generators import batch
from util.functions import trace
from util.vocabulary import Vocabulary

if USE_GPU:
  import cupy as np
else:
  import numpy as np

# Shift-reduce Operations
OP_SHIFT = 0
OP_UNARY = 1
OP_BINARY = 2
OP_FINISH = 3
NUM_OP = 4

def parse_args():
  def_vocab = 32768
  def_embed = 1024
  def_queue = 64
  def_stack = 64
  def_epoch = 20
  def_minibatch = 64
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
  p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
      help='vocabulary size (default: %d)' % def_vocab)
  p.add_argument('--embed', default=def_embed, metavar='INT', type=int,
      help='embedding layer size (default: %d)' % def_embed)
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
    if args.embed < 1: raise ValueError('you must set --embed >= 1')
    if args.queue < 1: raise ValueError('you must set --queue >= 1')
    if args.stack < 1: raise ValueError('you must set --stack >= 1')
    if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
    if args.minibatch < 1: raise ValueError('you must set --minibatch >= 1')
  except Exception as ex:
    p.print_usage(file=sys.stderr)
    print(ex, file=sys.stderr)
    sys.exit()

  return args

def my_zeros(shape, dtype):
  return Variable(np.zeros(shape, dtype=dtype))

def my_array(array, dtype):
  return Variable(np.array(array, dtype=dtype))

def binarize(tree):
  if isinstance(tree, Tree):
    children = []
    for ch in tree:
      rhs = binarize(ch)
      if len(children) == 2:
        lhs = Tree('@' + tree.label(), children)
        children = [lhs, rhs]
      else:
        children.append(rhs)
      
    return Tree(tree.label(), children)
  else:
    return tree

def make_tree(text):
  tree = Tree.fromstring(text)[0] # ignore ROOT
  return binarize(tree)

def extract_words(tree):
  if isinstance(tree, Tree):
    ret = []
    for ch in tree:
      ret += extract_words(ch)
    return ret
  else:
    return [tree]

def extract_phrase_labels(tree):
  if isinstance(tree, Tree) and isinstance(tree[0], Tree):
    ret = [tree.label()]
    for ch in tree:
      ret += extract_phrase_labels(ch)
    return ret
  else:
    return []

def extract_semi_labels(tree):
  if isinstance(tree, Tree):
    if isinstance(tree[0], Tree):
      ret = []
      for ch in tree:
        ret += extract_semi_labels(ch)
      return ret
    else:
      return [tree.label()]
  else:
    return []

def make_operations(tree):
  def recursive(tree):
    if isinstance(tree, Tree):
      if len(tree) == 1:
        if isinstance(tree[0], Tree):
          # unary
          return recursive(tree[0]) + [(OP_UNARY, tree.label())]
        else:
          # semiterminal
          return [(OP_SHIFT, tree.label())]
      elif len(tree) == 2 and isinstance(tree[0], Tree) and isinstance(tree[1], Tree):
        # binary
        return recursive(tree[0]) + recursive(tree[1]) + [(OP_BINARY, tree.label())]
    raise RuntimeError('invalid tree format')
  return recursive(tree) + [(OP_FINISH, None)]

def restore_labels(tree, phrase_vocab, semi_vocab):
  if isinstance(tree, Tree):
    if isinstance(tree[0], Tree):
      label = phrase_vocab.itos(tree.label())
    else:
      label = semi_vocab.itos(tree.label())
    children = [restore_labels(ch, phrase_vocab, semi_vocab) for ch in tree]
    return Tree(label, children)
  else:
    return tree

def convert_word_list(word_list, word_vocab):
  return [(text, word_vocab.stoi(text)) for text in word_list]

def convert_op_list(op_list, phrase_vocab, semi_vocab):
  ret = []
  for op, label in op_list:
    if op == OP_SHIFT:
      label = semi_vocab.stoi(label)
    elif op == OP_FINISH:
      label = None
    else:
      label = phrase_vocab.stoi(label)
    ret.append((op, label))
  return ret

def combine_xbar(tree):
  if isinstance(tree, Tree):
    children = [combine_xbar(ch) for ch in tree]
    label = tree.label()
    while children and isinstance(children[0], Tree) and children[0].label() == '@' + label:
      children = [ch for ch in children[0]] + children[1:]
    return Tree(label, children)
  else:
    return tree

def tree_to_string(tree):
  if isinstance(tree, Tree):
    ret = '(' + str(tree.label())
    for ch in tree:
      ret += ' ' + tree_to_string(ch)
    return ret + ')'
  else:
    return str(tree)

class WordEmbedding(Chain):
  def __init__(self, n_vocab, n_embed):
    super(WordEmbedding, self).__init__(
        w_xy = links.EmbedID(n_vocab, n_embed),
    )

  def __call__(self, x):
    return functions.tanh(self.w_xy(x))

class LinearEncoder(Chain):
  def __init__(self, n_input, n_output):
    super(LinearEncoder, self).__init__(
        w_xy = links.Linear(n_input, 4 * n_output),
        w_yy = links.Linear(n_output, 4 * n_output),
    )

  def __call__(self, c, x, y):
    return functions.lstm(c, self.w_xy(x) + self.w_yy(y))

class OperationEstimator(Chain):
  def __init__(self, n_embed, n_queue_state, n_stack_state):
    super(OperationEstimator, self).__init__(
        w_xo = links.Linear(n_embed, NUM_OP),
        w_qo = links.Linear(n_queue_state, NUM_OP),
        w_s1o = links.Linear(n_stack_state, NUM_OP),
        w_s2o = links.Linear(n_stack_state, NUM_OP),
        w_s3o = links.Linear(n_stack_state, NUM_OP),
    )

  def __call__(self, x, q, s1, s2, s3):
    return self.w_xo(x) + self.w_qo(q) + self.w_s1o(s1) + self.w_s2o(s2) + self.w_s3o(s3)

class ShiftTransition(Chain):
  def __init__(self, n_embed, n_queue_state, n_stack_state):
    super(ShiftTransition, self).__init__(
        w_xs = links.Linear(n_embed, n_stack_state),
        w_qs = links.Linear(n_queue_state, n_stack_state),
        w_s1s = links.Linear(n_stack_state, n_stack_state),
    )

  def __call__(self, x, q, s1):
    return functions.tanh(
        self.w_xs(x) + self.w_qs(q) + self.w_s1s(s1)
    )

class UnaryTransition(Chain):
  def __init__(self, n_queue_state, n_stack_state):
    super(UnaryTransition, self).__init__(
        w_qs = links.Linear(n_queue_state, n_stack_state),
        w_s1s = links.Linear(n_stack_state, n_stack_state),
        w_s2s = links.Linear(n_stack_state, n_stack_state),
    )

  def __call__(self, q, s1, s2):
    return functions.tanh(
        self.w_qs(q) + self.w_s1s(s1) + self.w_s2s(s2)
    )

class BinaryTransition(Chain):
  def __init__(self, n_queue_state, n_stack_state):
    super(BinaryTransition, self).__init__(
        w_qs = links.Linear(n_queue_state, n_stack_state),
        w_s1s = links.Linear(n_stack_state, n_stack_state),
        w_s2s = links.Linear(n_stack_state, n_stack_state),
        w_s3s = links.Linear(n_stack_state, n_stack_state),
    )

  def __call__(self, q, s1, s2, s3):
    return functions.tanh(
        self.w_qs(q) + self.w_s1s(s1) + self.w_s2s(s2) + self.w_s3s(s3)
    )

class LabelEstimator(Chain):
  def __init__(self, n_input, n_output):
    super(LabelEstimator, self).__init__(
        w_xy = links.Linear(n_input, n_output)
    )

  def __call__(self, x):
    return self.w_xy(x)
    
class Parser(Chain):
  def __init__(self,
      n_vocab, n_embed, n_queue_state, n_stack_state,
      n_phrase_label, n_semi_label):
    super(Parser, self).__init__(
        net_embed = WordEmbedding(n_vocab, n_embed),
        net_encoder = LinearEncoder(n_embed, n_queue_state),
        net_operation = OperationEstimator(n_embed, n_queue_state, n_stack_state),
        net_shift = ShiftTransition(n_embed, n_queue_state, n_stack_state),
        net_unary = UnaryTransition(n_queue_state, n_stack_state),
        net_binary = BinaryTransition(n_queue_state, n_stack_state),
        net_phrase_label = LabelEstimator(n_stack_state, n_phrase_label),
        net_semi_label = LabelEstimator(n_stack_state, n_semi_label),
    )
    self.n_vocab = n_vocab
    self.n_embed = n_embed
    self.n_queue_state = n_queue_state
    self.n_stack_state = n_stack_state
    self.n_phrase_label = n_phrase_label
    self.n_semi_label = n_semi_label

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.n_vocab, file=fp)
      print(self.n_embed, file=fp)
      print(self.n_queue_state, file=fp)
      print(self.n_stack_state, file=fp)
      print(self.n_phrase_label, file=fp)
      print(self.n_semi_label, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      n_vocab = int(next(fp))
      n_embed = int(next(fp))
      n_queue_state = int(next(fp))
      n_stack_state = int(next(fp))
      n_phrase_label = int(next(fp))
      n_semi_label = int(next(fp))
      return Parser(
          n_vocab, n_embed, n_queue_state, n_stack_state,
          n_phrase_label, n_semi_label,
      )

  def forward(self, word_list, op_list, unary_limit):
    is_training = op_list is not None

    # check args
    if len(word_list) < 1:
      raise ValueError('Word list is empty.')
    if is_training:
      n_shift = 0
      n_binary = 0
      for op, _ in op_list:
        if op == OP_SHIFT: n_shift += 1
        if op == OP_BINARY: n_binary += 1
      if n_shift != len(word_list) or n_binary != len(word_list) - 1:
        raise ValueError(
            'Invalid operation number: SHIFT=%d (required: %d), BINARY=%d (required: %d)' %
            (n_shift, n_binary, len(word_list), len(word_list) - 1))
      if op_list[-1] != (OP_FINISH, None):
        raise ValueError('Last operation is not OP_FINISH.')

    # initial values
    EMBED_ZEROS = my_zeros((1, self.n_embed), np.float32)
    QUEUE_ZEROS = my_zeros((1, self.n_queue_state), np.float32)
    STACK_ZEROS = my_zeros((1, self.n_stack_state), np.float32)
    NEG_INF = -1e20

    # queue encoding
    xq_list = []
    c = QUEUE_ZEROS
    q = QUEUE_ZEROS
    for text, wid in reversed(word_list):
      x = self.net_embed(my_array([wid], np.int32))
      c, q = self.net_encoder(c, x, q)
      xq_list.insert(0, (text, x, q))

    s_list = []
    unary_chain = 0
    if is_training:
      loss = my_zeros((), np.float32)

    # estimate
    for i in itertools.count():
      text, x, q = xq_list[0] if xq_list else ('', EMBED_ZEROS, QUEUE_ZEROS)
      t1, s1 = s_list[-1] if s_list else (None, STACK_ZEROS)
      t2, s2 = s_list[-2] if len(s_list) > 1 else (None, STACK_ZEROS)
      t3, s3 = s_list[-3] if len(s_list) > 2 else (None, STACK_ZEROS)

      op = self.net_operation(x, q, s1, s2, s3)

      if is_training:
        loss += functions.softmax_cross_entropy(op, my_array([op_list[i][0]], np.int32))
        op_argmax = op_list[i][0]
      else:
        op_filter = [0.0 for _ in range(NUM_OP)]
        filtered = 0
        if not xq_list:
          op_filter[OP_SHIFT] = NEG_INF
          filtered += 1
        if not s_list or unary_chain >= unary_limit:
          op_filter[OP_UNARY] = NEG_INF
          filtered += 1
        if len(s_list) < 2:
          op_filter[OP_BINARY] = NEG_INF
          filtered += 1
        if xq_list or len(s_list) > 1:
          op_filter[OP_FINISH] = NEG_INF
        if filtered == NUM_OP:
          raise RuntimeError('No possible operation!')

        op += my_array([op_filter], np.float32)
        op_argmax = int(cuda.to_cpu(op.data.argmax(1)))

      if op_argmax == OP_SHIFT:
        t0 = Tree(None, [text])
        s0 = self.net_shift(x, q, s1)
        xq_list.pop(0)
        unary_chain = 0
        label = self.net_semi_label(s0)
      elif op_argmax == OP_UNARY:
        t0 = Tree(None, [t1])
        s0 = self.net_unary(q, s1, s2)
        s_list.pop()
        unary_chain += 1
        label = self.net_phrase_label(s0)
      elif op_argmax == OP_BINARY:
        t0 = Tree(None, [t2, t1])
        s0 = self.net_binary(q, s1, s2, s3)
        s_list.pop()
        s_list.pop()
        unary_chain = 0
        label = self.net_phrase_label(s0)
      else: # OP_FINISH
        break

      if is_training:
        loss += functions.softmax_cross_entropy(label, my_array([op_list[i][1]], np.int32))
        label_argmax = op_list[i][1]
      else:
        label_argmax = int(cuda.to_cpu(label.data.argmax(1)))

      t0.set_label(label_argmax)
      s_list.append((t0, s0))

      '''
      if is_training:
        op_est = int(cuda.to_cpu(op.data.argmax(1)))
        label_est = int(cuda.to_cpu(label.data.argmax(1)))
        trace('%c %c gold=%d-%2d, est=%d-%2d, stack=%2d, queue=%2d' % (
            '*' if op_est == op_list[i][0] else ' ',
            '*' if label_est == op_list[i][1] else ' ',
            op_list[i][0], op_list[i][1],
            op_est, label_est,
            len(s_list), len(xq_list)))
      '''

    if is_training:
      return loss
    else:
      # combine multiple trees if they exists, and return the result.
      t0, _ = s_list.pop()
      if s_list:
        raise RuntimeError('There exist multiple subtrees!')
      return t0

def train(args):
  trace('loading corpus ...')
  with open(args.source) as fp:
    trees = [make_tree(l) for l in fp]

  trace('extracting leaf nodes ...')
  word_lists = [extract_words(t) for t in trees]

  trace('extracting gold operations ...')
  op_lists = [make_operations(t) for t in trees]

  trace('making vocabulary ...')
  word_vocab = Vocabulary.new(word_lists, args.vocab)
  phrase_set = set()
  semi_set = set()
  for tree in trees:
    phrase_set |= set(extract_phrase_labels(tree))
    semi_set |= set(extract_semi_labels(tree))
  phrase_vocab = Vocabulary.new([list(phrase_set)], len(phrase_set), add_special_tokens=False)
  semi_vocab = Vocabulary.new([list(semi_set)], len(semi_set), add_special_tokens=False)

  trace('converting data ...')
  word_lists = [convert_word_list(x, word_vocab) for x in word_lists]
  op_lists = [convert_op_list(x, phrase_vocab, semi_vocab) for x in op_lists]

  trace('start training ...')
  parser = Parser(
      args.vocab, args.embed, args.queue, args.stack,
      len(phrase_set), len(semi_set),
  )
  if USE_GPU:
    parser.to_gpu()
  opt = optimizers.AdaGrad(lr = 0.005)
  opt.setup(parser)
  opt.add_hook(optimizer.GradientClipping(5))

  for epoch in range(args.epoch):
    n = 0
    
    for samples in batch(zip(word_lists, op_lists), args.minibatch):
      parser.zerograds()
      loss = my_zeros((), np.float32)

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
    semi_vocab.save(prefix + '.semiterminals')
    parser.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', parser)

  trace('finished.')

def test(args):
  trace('loading model ...')
  word_vocab = Vocabulary.load(args.model + '.words')
  phrase_vocab = Vocabulary.load(args.model + '.phrases')
  semi_vocab = Vocabulary.load(args.model + '.semiterminals')
  parser = Parser.load_spec(args.model + '.spec')
  if USE_GPU:
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
              semi_vocab))
      print('( ' + tree_to_string(tree) + ' )')

  trace('finished.')

def main():
  args = parse_args()
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)

if __name__ == '__main__':
  main()

