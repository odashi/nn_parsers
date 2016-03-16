import sys
import datetime

__start = None

def trace(*args):
  global __start
  if __start is None:
    __start = datetime.datetime.now()
  print(datetime.datetime.now() - __start, '...', *args, file=sys.stderr)
  sys.stderr.flush()

def fill_batch(batch, token='</s>'):
  max_len = max(len(x) for x in batch)
  return [x + [token] * (max_len - len(x) + 1) for x in batch]

def fill_batch2(batch, start_token='<s>', end_token='</s>'):
  max_len = max(len(x) for x in batch)
  return [[start_token] + x + [end_token] * (max_len - len(x) + 1) for x in batch]

