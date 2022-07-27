import os
import unicodedata
import re
from io import open
import string
import random
import torch

class Lang:
  def __init__(self, name):
    '''
    name(str): Language's name (ex. eng, fra)
    '''
    self.name = name
    self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2}
    self.word2count = {"<SOS>": 0, "<EOS>": 0, "<PAD>": 0}
    self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>"}
    self.n_words = 3

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse = False):
  '''
  open txt file and return each of Lang class, pair lists
  inputs
    - lang1(str): language's name
    - lang2(str): language's name
    - reverse(bool): lang2 -> lang1 if reverse else lang1 -> lang2 (->: translation) 
  returns
    - input_lang(class)
    - output_lang(class)
    - pairs(list): [lang1 sentence, lang2 sentence] * (n_sentences) in txt file
  '''

  print('Reading lines...')
  with open(os.path.join(os.getcwd(), 'data/%s-%s.txt'%(lang1, lang2)), encoding = 'utf-8') as f:
    lines = f.read().strip().split('\n')

  pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

  if reverse:
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)
  else:
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

  return input_lang, output_lang, pairs

# MAX_LENGTH = 10
'''
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
 )
'''

def filterPair(p, max_length, eng_prefixes):
  '''
  filter pair(in/output sentence)
  1. n_tokens of pair < (MAX_LENGTH - 2) for <SOS>, <EOS> token
  2. output sentence(eng) starts with first pharse in eng_prefixes 
  input:
    - p(list): [intput sentence, output sentence]
    - max_length(int)
    - eng_prefixes(list)
  output:
    - filtered pair(list): [input sentence, output sentence]
  '''
  return len(p[0].split(' ')) < (max_length - 2) and \
      len(p[1].split(' ')) < (max_length - 2)and \
      p[1].startswith(eng_prefixes)


def filterPairs(pairs, max_length, eng_prefixes):
  '''
  filter all pairs 
  '''
  return [pair for pair in pairs if filterPair(pair, max_length, eng_prefixes)]

def prepareData(lang1, lang2, max_length, eng_prefixes, reverse = False, random_shuffle = False):
  '''
  inputs:
    - lang1(str): input language type (ex. eng)
    - lang2(str): output language type (ex. fra)
    - max_length(int): max length of tokens in each sentence
    - reverse(bool): lang2 -> lang1 if reverse else lang1 -> lang2 (->: translation) 
    - random_shuffle(bool): sentence(dataset) will be shuffled randomly
  outputs:
    - input_lang(class): class Lang()
    - output_lang(class): class Lang()
    - pairs(list): [input sentence, output sentence]
  '''

  input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
  print('Read %s sentence pairs' % len(pairs))
  pairs = filterPairs(pairs, max_length, eng_prefixes)
  # print('Trimmed to %s sentence pairs' %(pairs))
  print('Counting words...')

  # for parallel computation use pad(fix every sentences' length as max_length)
  for pair in pairs:
    pair[0] = '<SOS> ' + pair[0] 
    pair[1] = '<SOS> ' + pair[1] 
    pair[0] += ' <EOS>'
    pair[1] += ' <EOS>'
    if len(pair[0].split(' ')) < max_length:
      n_pad = max_length - len(pair[0].split(' ')) 
      pair[0] += ' <PAD>' * n_pad
    if len(pair[1].split(' ')) < max_length:
      n_pad = max_length - len(pair[1].split(' '))
      pair[1] += ' <PAD>' * n_pad  
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])

  print('Counted words:')
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  if random_shuffle:
    random.shuffle(pairs)
  return input_lang, output_lang, pairs

# input_lang, output_lang, pairs = prepareData('eng', 'fra', MAX_LENGTH, True, True)
# print(random.choice(pairs))

def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, device):
  indexes = indexesFromSentence(lang, sentence)
  # indexes.append(EOS_token)
  return torch.tensor(indexes, dtype = torch.long, device = device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair, device):
  input_tensor = tensorFromSentence(input_lang, pair[0], device)
  target_tensor = tensorFromSentence(output_lang, pair[1], device)
  return (input_tensor, target_tensor)

