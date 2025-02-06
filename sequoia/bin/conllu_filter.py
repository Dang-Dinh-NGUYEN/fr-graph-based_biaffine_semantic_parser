#!/usr/bin/env python3
"""
Filter a corpus based on a sent_id list. Script used to produced train/dev/test 
split from UD release. Two arguments are required:
* arg 1: a file which contains the sent_id list (on sent_id by line)
* arg 2: the input corpus
Output is written on stdout, user should redirect to appropriate file.

Script from Sequoia tools git repo (by Bruno Guillaume, I guess).
Adapted by Carlos Ramisch to work with conllu-plus files (more than 10 columns)
"""
import sys
import conllu
import pdb

if len(sys.argv) != 3:
  print ('need two args: sent_id list, 1 per line + input corpus')
  exit (1)

with open(sys.argv[1], "r", encoding="UTF-8") as f:
  ids = f.readlines()

with open(sys.argv[2], "r", encoding="UTF=8") as f:    
  in_corpus_dict = {}
  for sent in conllu.parse_incr(f):
    in_corpus_dict[sent.metadata['sent_id']] = sent
    if sent.metadata.get("global.columns",None):      
      header = sent.metadata["global.columns"] # save header for later
      del sent.metadata["global.columns"] # remove it from sentence metadata
      
#pdb.set_trace()      
selected = [in_corpus_dict.get(sent_id.strip(), None) for sent_id in ids]

print("# global.columns = " + header)  

for sent in selected:
  if sent : # Ignore "None" - absent sentences
    print(sent.serialize(), end="") 

