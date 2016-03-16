from collections import defaultdict


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
  def new(list_generator, size, add_special_tokens=True):
    self = Vocabulary()
    self.__size = size

    if add_special_tokens:
      SPECIAL_TOKENS = ['<unk>', '<s>', '</s>']
    else:
      SPECIAL_TOKENS = []
    NUM_ST = len(SPECIAL_TOKENS)
    
    word_freq = defaultdict(lambda: 0)
    for words in list_generator:
      for word in words:
        word_freq[word] += 1

    self.__stoi = defaultdict(lambda: 0)
    self.__itos = [''] * self.__size

    for i, tok in enumerate(SPECIAL_TOKENS):
      self.__stoi[tok] = i
      self.__itos[i] = tok
    
    for i, (k, v) in zip(range(self.__size - NUM_ST), sorted(word_freq.items(), key=lambda x: -x[1])):
      self.__stoi[k] = i + NUM_ST
      self.__itos[i + NUM_ST] = k

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

