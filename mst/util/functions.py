import datetime
import sys

__start = None
def trace(*args, rollback=False):
  global __start
  if __start is None:
    __start = datetime.datetime.now()
  print( \
    datetime.datetime.now() - __start, \
    '...', \
    *args, \
    file=sys.stderr, \
    end='\r' if rollback else '\n',
  )
  sys.stderr.flush()

