# About the Sequoia corpus

## Overview
Reminding that the primary objective of this project is to implement a graph-based biaffine semantic parser of French, 
the corpora must be carefully selected to ensure they accurately represent the complexity and diversity of the French 
language, thereby providing a robust foundation for training and evaluating the parser on realistic semantic structures.

The Sequoia corpora, with its multiple versions, context richness and high-quality syntactic annotations of French sentences[^sequoia], has been
selected for our project.

### Universal dependencies
Universal dependencies (UD) is a project developing cross-linguistically consistent treebank annotation for many languages,
with goal of facilitating multilingual parser development, cross-lingual learning, and parsing research from a language
typology perspective. The general philosophy is to provide a universal inventory of categories and guidelines to facilitate 
consistent annotation of similar constructions across languages, while allowing language-specific extensions when necessary[^UD].

### Sequoia annotated corpus for french

#### TO BE COMPLETED (differences between version of Sequoia corpora)

### Leveraging the Sequoia Corpus for graph-based biaffine dependency parsing[^carlosRAMISCH]

---
#### Source of the corpus

In this project, we use the latest version of the Sequoia corpus, which was downloaded from the Deep Sequoia's git repository:
[Deep Sequoia repository](https://gitlab.inria.fr/sequoia/deep-sequoia). This version is also known as the "Sequoia-UD_SPF"
containing Surface, Parseme and FrSemCor layers

We obtained the file trunk/sequoia-ud.parseme.frsemcor from commit number ea758e94
The file is the result of the conversion from Sequoia's source as described on the [documentation](https://deep-sequoia.inria.fr/process/).

We keep the original file in `./sequoia/src` folder to make command line completion faster

---

#### Simplification
We run `simplify_sequoia.py` to make the file easier to work with. 

Look at the header of the script to understand the simplifications carried out.
Justifications for each simplification are indicated below their description with an arrow "=>"
Here is the log of the simplifications:

```
./bin/simplify_sequoia.py src/sequoia-ud.parseme.frsemcor > sequoia-ud.parseme.frsemcor.simple.full

1952 range tokens removed.

416 discontinuous and overlapping NEs removed.

6578 supersense tags removed (on MWEs or strange POS).
1282 supersense tags modified (complex operators).

66 non-projective sentences removed:
annodis.er_00018, annodis.er_00026, annodis.er_00043, annodis.er_00079, annodis.er_00092, annodis.er_00099, annodis.er_00195, annodis.er_00204, annodis.er_00214, annodis.er_00218, annodis.er_00267, annodis.er_00357, annodis.er_00386, annodis.er_00400, annodis.er_00410, annodis.er_00448, annodis.er_00473, annodis.er_00475, annodis.er_00477, annodis.er_00519, emea-fr-dev_00024, emea-fr-dev_00044, emea-fr-test_00043, emea-fr-test_00207, emea-fr-test_00274, emea-fr-test_00472, emea-fr-test_00499, Europar.550_00004, Europar.550_00005, Europar.550_00029, Europar.550_00041, Europar.550_00043, Europar.550_00209, Europar.550_00257, Europar.550_00276, Europar.550_00289, Europar.550_00295, Europar.550_00303, Europar.550_00307, Europar.550_00335, Europar.550_00355, Europar.550_00367, Europar.550_00473, Europar.550_00476, Europar.550_00488, Europar.550_00518, Europar.550_00544, frwiki_50.1000_00066, frwiki_50.1000_00087, frwiki_50.1000_00091, frwiki_50.1000_00124, frwiki_50.1000_00137, frwiki_50.1000_00215, frwiki_50.1000_00246, frwiki_50.1000_00305, frwiki_50.1000_00318, frwiki_50.1000_00381, frwiki_50.1000_00426, frwiki_50.1000_00455, frwiki_50.1000_00523, frwiki_50.1000_00565, frwiki_50.1000_00621, frwiki_50.1000_00655, frwiki_50.1000_00829, frwiki_50.1000_00843, frwiki_50.1000_00867
```

### Splitting

The files are then split into train, dev and test according to the IDs, following the official UD release.
We got the IDs from [the tools folder](https://gitlab.inria.fr/sequoia/deep-sequoia/tree/master/tools)
The script `conllu_filter.py` was adapted to consider 

```
bin/conllu_filter.py bin/train.ids sequoia-ud.parseme.frsemcor.simple.full > sequoia-ud.parseme.frsemcor.simple.train
bin/conllu_filter.py bin/dev.ids   sequoia-ud.parseme.frsemcor.simple.full > sequoia-ud.parseme.frsemcor.simple.dev
bin/conllu_filter.py bin/test.ids  sequoia-ud.parseme.frsemcor.simple.full > sequoia-ud.parseme.frsemcor.simple.test
```

We also generate a toy small version of the corpus that can be useful to test your code during development:
```
NBSENT=30
CORPUS=sequoia-ud.parseme.frsemcor.simple.dev
LASTID=`grep "sent_id" ${CORPUS} | head -n  $((NBSENT+1)) | tail -n 1 | sed 's/.* \([^ ]*\)$/\1/g'`
CUTLINE=`grep -n ${LASTID} ${CORPUS} | sed 's/:.*//g'`
head -n $((CUTLINE-1)) ${CORPUS} > sequoia-ud.parseme.frsemcor.simple.small
```

The file `tiny.conllu` was manually extracted and simplified, it is used in parsing exercises.

Finally, we also split the non-simplified version of the corpus into train, dev and test (before simplification).
These files should not be used in your experiments.
```
bin/conllu_filter.py bin/train.ids src/sequoia-ud.parseme.frsemcor > src/sequoia-ud.parseme.frsemcor.simple.train
bin/conllu_filter.py bin/dev.ids   src/sequoia-ud.parseme.frsemcor > src/sequoia-ud.parseme.frsemcor.simple.dev
bin/conllu_filter.py bin/test.ids  src/sequoia-ud.parseme.frsemcor > src/sequoia-ud.parseme.frsemcor.simple.test
```

#### Usage
For the earlier step of the project, we use the simplified version of the Sequoia corpus to facilitate the implementation
process. Later, we plan to revisit the original version to adapt of projet to the task of predicting arbitrary graphs.

### References

[^sequoia] : [Marie Candito, Djamé Seddah. Le corpus Sequoia : annotation syntaxique et exploitation pour l’adaptation d’analyseur par pont lexical. TALN 2012 - 19e conférence sur le Traitement Automatique des Langues Naturelles, Jun 2012, Grenoble, France. hal-00698938](https://inria.hal.science/hal-00698938v1/document)

[^UD] : [https://universaldependencies.org/introduction.html](https://universaldependencies.org/introduction.html)

[^carlosRAMISCH] : [https://gitlab.lis-lab.fr/carlos.ramisch/pstal-etu](https://gitlab.lis-lab.fr/carlos.ramisch/pstal-etu)