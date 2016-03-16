from nltk.tree import Tree

# Shift-reduce Operations
OP_SHIFT = 0
OP_UNARY = 1
OP_BINARY = 2
OP_FINISH = 3
NUM_OP = 4

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

def extract_semiterminals(tree):
  if isinstance(tree, Tree):
    if isinstance(tree[0], Tree):
      ret = []
      for ch in tree:
        ret += extract_semiterminals(ch)
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

def convert_word_list(word_list, word_vocab):
  return [(w, word_vocab.stoi(w.lower())) for w in word_list]

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
  def recursive(tree):
    if isinstance(tree, Tree):
      xbar = tree.label()[0] == '@'
      label = tree.label()[1:] if xbar else tree.label()
      children = []
      for ch in tree:
        if isinstance(ch, Tree):
          ch, ch_xbar = recursive(ch)
          if ch_xbar and ch.label() == label:
            children += [gch for gch in ch]
          else:
            children.append(ch)
        else:
          children.append(ch)
      return Tree(label, children), xbar
    else:
      return tree, False
  return recursive(tree)[0]

def tree_to_string(tree):
  if isinstance(tree, Tree):
    ret = '(' + str(tree.label())
    for ch in tree:
      ret += ' ' + tree_to_string(ch)
    return ret + ')'
  else:
    return str(tree)

