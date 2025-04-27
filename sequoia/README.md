# Deep Sequoia corpus for semantic parsing

This repository contains different versions of the [Deep Sequoia corpus](https://gitlab.inria.fr/sequoia/deep-sequoia),
tailored for academic use — specifically within the Structured Prediction in NLP course of the Master IAAA program at 
Aix-Marseille University and Ecole Centrale Méditerranée.

It includes key information on corpus versions and outlines preprocessing steps, with a particular focus on the 
```sequoia.deep_and_surf.parseme.frsemcor``` version.

## sequoia-ud.parseme.frsemcor
> See [this link](https://gitlab.lis-lab.fr/carlos.ramisch/pstal-etu/-/tree/master/sequoia?ref_type=heads) for detailed information and preprocessing steps.

## sequoia.deep_and_surf.parseme.frsemcor
### Source of the version
The corpus was downloaded from the [Deep Sequoia's git repository](https://gitlab.inria.fr/sequoia/deep-sequoia/-/tree/master/tags/sequoia-9.2?ref_type=heads)
(tags/sequoia-9.2/). 

We obtained the most recent version of ```sequoia.deep_and_surf.parseme.frsemcor``` from the commit ```3cca2cc3```. This
version of Sequoia contains both surface syntactic and deep semantic annotations, combined with PARSEME-FR MWE annotation format and 
FRSEMCOR semantic annotation format. Therefore, it can be used for both syntactic and semantic parsing tasks.

### Format documentation
The format is an extended version of the CoNLL-U format:

- columns 1 to 10 encode morphology and syntax in a CoNLL-U inspired format, adapted to encode graphs
-- the syntactic columns (7 - HEAD and 8 - DEPREL) encode both the surface dependency tree and the deep syntactic graph
- column 11 encodes the PARSEME-FR annotation of named entities and multi-word expressions
- column 12 contains the FRSEMCOR semantic annotation on nouns

### Deep-and-surf dependency format
The deep-and-surf format is a compact representation containing BOTH the (surface) dependency tree
and the deep dependency graph.

For each token ```t```, columns 7 and 8 encode one or several arcs in which ```t``` is the dependent:
this is represented by column 7 containing several labels, and column 8 containing several governor ids.

The number of labels in column 7 is the same as the number of governors in column 8 and they are interpreted in parallel:
first label in column 7 corresponds to first governor in column 8, etc.

For instance, the noun ```réunion``` in the example below depends on tokens 5 and 2, with labels ```suj:obj``` and ```D:suj:obj``` respectively.

```
# sent_id = annodis.er_00449
# text = Une prochaine réunion est prévue mardi
1	Une	un	D	DET	g=f|n=s|s=ind	3	det	_	_
2	prochaine	prochain	A	ADJ	g=f|n=s|s=qual	3	mod	_	_
3	réunion	réunion	N	NC	g=f|n=s|s=c	5|2	suj:obj|D:suj:suj	_	_
4	est	être	V	V	dl=être|m=ind|n=s|p=3|t=pst|void=y	5	S:aux.pass	_	_
5	prévue	prévoir	V	VPP	diat=passif|dl=prévoir|dm=ind|g=f|m=part|n=s|t=past	0	root	_	_
6	mardi	mardi	N	NC	g=m|n=s|s=c	5	mod	_	_
```

A given arc belongs to either:
- The surface tree only (**prefix:** ```S:```)
- The deep syntactic graph only (**prefix:** ```D:```)
- Or to both (**no prefix**)

> relations subject to diathesis alternations contain a double label ```xxx:yyy```, where ```xxx``` stands for the "final" grammatical 
function, and ```yyy``` stands for the canonical grammatical function. 

**Warning: The annotation scheme of ```sequoia.deep_and_surf.parseme.frsemcor``` is inspired by CoNLL-U but does not fully 
comply with the Universal Dependencies framework (e.g., it uses multiple HEAD/DEPREL entries instead of DEPS, and employs
non-standard label names, etc.).**

**Note: There is currently no official solution for converting ```sequoia.deep_and_surf.parseme.frsemcor``` into a fully 
UD-compliant format. The only Sequoia version aligned with UD standards was produced externally by the 
[Surface Syntactic Universal Dependencies (SUD)](https://surfacesyntacticud.github.io/) initiative.
Since altering the annotation scheme could lead to different syntactic relations between words, such transformation falls
outside the scope of this project. Therefore, we focus solely on preprocessing the corpus as outlined below.**

### Obtaining the deep syntactic graphs
The deep format is obtained by removing all relations prefixed with ```S:``` (except when they are the only relation) and by 
extracting the canonical function, i.e., the second part, from double-labeled dependencies.

We run ```simplify_sequoia.py``` to preprocess the file. This script allows us to remove range tokens, remove discontinuous 
and overlapping NEs, remove/modify supersense tags, remove non-projective sentences (for simple syntactic parsing) or 
obtain the deep syntactic graphs (for semantic parsing) and full-fill the DEPS column.

```
python sequoia/bin/simplify_sequoia.py sequoia/src/sequoia.deep_and_surf.parseme.frsemcor --deep_syntax > sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.full

0 range tokens removed.

0 discontinuous and overlapping NEs removed.

0 supersense tags removed (on MWEs or strange POS).
0 supersense tags modified (complex operators).

7417 surface relations removed .
9717 surface relations modified (surface-only relations).

0 non-projective sentences removed:
```

**Note: This script is compatible with both versions of the corpus available in this repository.**

### Splitting
The files are then split into train, dev and test according to the IDs, following the official UD release.
We got the IDs from [the tools folder](https://gitlab.inria.fr/sequoia/deep-sequoia/tree/master/tools).

The script ```conllu_filter.py``` was adapted to consider
```
sequoia/bin/conllu_filter.py sequoia/bin/train.ids sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.full > sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.train
sequoia/bin/conllu_filter.py sequoia/bin/dev.ids sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.full > sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.dev
sequoia/bin/conllu_filter.py sequoia/bin/test.ids sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.full > sequoia/sequoia.deep_and_surf.parseme.frsemcor.simple.test
```