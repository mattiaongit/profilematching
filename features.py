from  __future__ import division
from functools import reduce, wraps
import math
import string
from stats import distribution
from keyboard import *

alphabet = string.ascii_lowercase

def candidatepriors(f):
  @wraps(f)
  def _f(*args,**kwargs):
    output = []
    output.extend(f(args[0][0],**kwargs))

    priors_data = []
    priors = args[1]
    for p in priors:
        priors_data.append(f(p,**kwargs))
    column_data = zip(*priors_data)
    [output.extend(distribution(d)) for d in column_data]
    return output
  return _f


### METRICS FUNCTIONS ###
### PATTERNS DUE TO HUMAN LIMATIONS  Time/Memory###

# Same username
def sameUsername(candidate, priors):
    return [priors.count(candidate)]

# Username lenght likelihood
def ull(candidate, priors):
  output = [len(candidate)]
  output.extend(distribution([len(p) for p in priors]))
  return output

# Unique username creation likelihood
def uucl(candidate, priors):
  return [len(set(priors)) / len(priors)]

### EXOGENOUS FACTORS ###
### TYPING PATTERNS ###

def biGrams(word):
  return [[word[x],word[x+1]] for x in range(0, len(word)-1)]

# keys is a pair of key e.g ('a','q')
# Return boolean or the string rapresenting the hand used
def sameHand(keys, handInfo = False, layout = 'qwerty'):
  lefthand,righthand = left_hand(layout), right_hand(layout)
  if not handInfo:
    return ( keys[0] in lefthand and keys[1] in lefthand ) or ( keys[0] in righthand and keys[1] in righthand )
  else:
    return ( keys[0] in lefthand and keys[1] in lefthand and 'left') or ( keys[0] in righthand and keys[1] in righthand and 'right')

def sameFinger(keys, layout = 'qwerty'):
  if sameHand(keys, layout = layout):
    samefinger = [all((keys[0] in finger, keys[1] in finger)) for finger in typing_map[sameHand(keys,True, layout)].values()]
    return sum(samefinger) > 0 and True or False
  else:
    return False

# The percentage of keys typed using the same (X) used for the previous key.
# (X) depending on the granularities e.g 'Hand' or 'Finger'
@candidatepriors
def sameRate(username, granularitiesFunction, layout = 'qwerty'):
  username = username.replace(" ","").lower()
  bigram = biGrams(username)
  samerate = [granularitiesFunction(bg, layout = layout) for bg in bigram]
  return (len(username) == 1 and [1]) or [sum(samerate) / (len(username) -1)]


# The percentage of keys typed using each finger order by hands order by finger (left-right/index,middle,pinkie,ring)

@candidatepriors
def eachFingerRate(username, layout = 'qwerty'):
  to_flat = [[(finger, hand, sum([username.count(key)
            for key in typing_map[layout][hand][finger]])/len(username))
            for finger in typing_map[layout][hand]]
            for hand in typing_map.keys()]
  ordered = sorted([rate for hand in to_flat for rate in hand], key = lambda tup: (tup[0],tup[1]))
  return [el[2] for el in ordered]


#The percentage of keys pressed on rows: Top Row, Home Row, Bottom Row, and Number Row
@candidatepriors
def rowsRate(username, layout = 'qwerty' ):
  return [sum([c in row for c in username]) for row in typing_row[layout]]



# The approximate distance (in meters) traveled for typing a username
# Normal typing keys are assumed to be (1.8cm)^2 (including gap between keys).
def travelledDistance(username):
  pass

### ENDOGENOUS FACTORS ###
@candidatepriors
def alphabetDistribution(username):
  return [username.count(c)/len(username) for c in alphabet]

@candidatepriors
def shannonEntropy(username):
  alphabetdstr = [username.count(c)/len(username) for c in alphabet]
  entropy = reduce((lambda x,y: x - (y * math.log(y,2) if y > 0 else 0)), alphabetdstr, 0)
  return [entropy]


def naivEntropy(text):
  text = set(text).intersection(set(alphabet))
  return [len(text) / len(alphabet)]


# Longest Common Substring - data is a collection of strings, eg : ['mattia','mattiadmr']
# If normalized return lcs lenght values in range [0,1] (normalized by the maximum length of the two\n strings)
# Usefull to catch prefixes - suffixes
def lcsubstring(candidate, priors):
  output = []
  for p in priors:
    data = (candidate, p)
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
      for i in range(len(data[0])):
        for j in range(len(data[0])-i+1):
          if j > len(substr) and all(data[0][i:i+j] in x for x in data):
            substr = data[0][i:i+j]
    output.extend([len(substr) / max([len(d) for d in data])])

  return distribution(output)

# Longest Common Subsequence
# Usefull to detect abbreviations
def lcs(candidate, priors):
  output = []
  for p in priors:
    data = (candidate, p)
    a = data[0]
    b = data[1]
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i, x in enumerate(a):
      for j, y in enumerate(b):
        if x == y:
          lengths[i+1][j+1] = lengths[i][j] + 1
        else:
          lengths[i+1][j+1] = \
            max(lengths[i+1][j], lengths[i][j+1])
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
      if lengths[x][y] == lengths[x-1][y]:
        x -= 1
      elif lengths[x][y] == lengths[x][y-1]:
        y -= 1
      else:
        assert a[x-1] == b[y-1]
        result = a[x-1] + result
        x -= 1
        y -= 1
    output.extend([len(result) / max([len(a),len(b)])])
  return distribution(output)

# Dynamic Time Warping
# TODO : How to apply this to strings? Alignment on time makes any sense on strings?
def dtw(data):
  pass
