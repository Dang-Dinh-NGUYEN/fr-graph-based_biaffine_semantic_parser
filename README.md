# Graph-based biaffine semantic parser of French

---
#### Author : NGUYEN Dang Dinh - M2 IAAA, Aix - Marseille Université
#### Supervisor : RAMISCH Carlos - TALEP (LIS - CNRS), Aix - Marseille Université

---
## About this project

---

This final project of our Master’s program in Artificial Intelligence and Machine Learning, delves into the core of 
Natural Language Processing (NLP) by developing a graph-based biaffine dependency parser that predicts labeled dependency 
structures. Unlike conventional tree-based approaches, our model is designed to predict arbitrary graphs, enabling it to
capture the complex, non-projective, and ambiguous syntactic relationships inherent in natural language.

Through rigorous implementation and evaluation, this project aims to advance the state-of-the-art in dependency parsing,
offering valuable insights into the strengths and limitations of graph-based models and paving the way for more robust 
and semantically aware language processing systems.

The project is mainly developed using python, pytorch, huggingface, and conllu libraries.

## Project organisation

---

## Libraries and dependencies
Before using this semantic parser, make sure that all necessary libraries and dependencies are installed by running the following command :

````shell
pip install -r requirements.txt
````

## Usage

---
### Syntactic parsing

---
### Prepare data

To prepare data from a given corpus, `./sequoia/sequoia-ud.parseme.frsemcor.simple.train` for example, and save
the output, run the following cell :

````shell
python ./run.py preprocess ./sequoia/sequoia-ud.parseme.frsemcor.simple.train -u -s=./sequoia/preprocessed_sequoia-ud.parseme.frsemcor.simple.train.pt
````

Data can also be preprocessed with existing vocabularies. The following shell performs the preprocessing on 
`./sequoia/sequoia-ud.parseme.frsemcor.simple.dev` with vocabularies extracted from the corpus `.train` : 

````shell
python ./run.py preprocess ./sequoia/sequoia-ud.parseme.frsemcor.simple.dev -l ./sequoia/preprocessed_sequoia-ud.parseme.frsemcor.simple.train.pt -s=./sequoia/preprocessed_sequoia-ud.parseme.frsemcor.simple.dev.pt
````

Other arguments are available as presented below :

````
usage: run.py preprocess [-h] [--columns COLUMNS] [--update] [--transformer {None,almanach/camembert-base}] [--max_len MAX_LEN] [--save] [--load LOAD] [--display] input_file

positional arguments:
  input_file            path to input file

options:
  -h, --help            show this help message and exit
  --columns, -c COLUMNS
                        column.s to be extracted (default = form, upos, head, deprel)
  --update, -u          update vocabulary during preprocessing
  --transformer {None,almanach/camembert-base}
                        name of pre_trained transformer to be used (default=None)
  --max_len, -m MAX_LEN
                        maximum sequence length (default=50)
  --save, -s            save preprocessed data
  --load, -l LOAD       path to preprocessed file
  --display, -d         display preprocessed data
````

**Note: By selecting a pre-trained transformer (e.g. almanach/camembert-base), contextual embeddings are preprocessed in addition.**

### Training model
At the beginning, we define our model and train it with similar configurations/conditions as mentioned in the article of 
Dozat and Manning [^biaffine]. 

````shell
python ./run.py train lstm --bidirectional -s ./models/sample_model.pkl
````
Other arguments are available as follows :
````
usage: run.py train [-h] [--ftrain FTRAIN] [--fdev FDEV] [--ltrain LTRAIN] [--ldev LDEV] [--d_arc D_ARC] [--d_rel D_REL] [--dropout_rate DROPOUT_RATE] [--n_epochs N_EPOCHS]
                    [--batch_size BATCH_SIZE] [--lr LR] [--patience PATIENCE] [--save SAVE]
                    {lstm,gru,almanach/camembert-base} ...

positional arguments:
  {lstm,gru,almanach/camembert-base}
                        encoder type: lstm | gru | almanach/camembert-base
    lstm                lstm encoder
    gru                 gru encoder
    almanach/camembert-base
                        camembert encoder

options:
  -h, --help            show this help message and exit
  --ftrain FTRAIN       path to train corpus
  --fdev FDEV           path to dev corpus
  --ltrain LTRAIN       path to preprocessed train file
  --ldev LDEV           path to preprocessed dev file
  --d_arc D_ARC         dimension of head/dependent vector (default=400)
  --d_rel D_REL         dimension of deprel vector (default=100)
  --dropout_rate, -r DROPOUT_RATE
                        dropout rate (default=0.33)
  --n_epochs N_EPOCHS   number of training epochs
  --batch_size BATCH_SIZE
                        batch_size
  --lr LR               learning rate
  --patience, -p PATIENCE
                        number of patiences
  --save, -s SAVE       path to save model
````

For RNN-based parser, our program allows the following options:

````
usage: run.py train lstm [-h] [--embeddings EMBEDDINGS] [--embeddings_dim EMBEDDINGS_DIM] [--d_h D_H] [--rnn_layers RNN_LAYERS] [--bidirectional]

options:
  -h, --help            show this help message and exit
  --embeddings, -e EMBEDDINGS
                        supplementary embeddings (default = form, upos)
  --embeddings_dim, -ed EMBEDDINGS_DIM
                        dimensions of supplementary embeddings (default = 100, 100)
  --d_h D_H             dimension of recurrent state (default=200)
  --rnn_layers RNN_LAYERS
                        number of rnn's layer (default=3)
  --bidirectional       enable bidirectional
````

The use of pre-trained transformer are supported with additional features:

```` 
usage: run.py train almanach/camembert-base [-h] [--embeddings EMBEDDINGS] [--embeddings_dim EMBEDDINGS_DIM] [--unfreeze UNFREEZE]

options:
  -h, --help            show this help message and exit
  --embeddings, -e EMBEDDINGS
                        supplementary embeddings
  --embeddings_dim, -ed EMBEDDINGS_DIM
                        dimensions of supplementary embeddings
  --unfreeze UNFREEZE   last n layers to be fine tuned (default=0)
````

**Note: Our a transformer-based parser was not designed to fine-tune any pre-trained transformer. Therefore, selecting --unfreeze has no effects!**

### Predict
To predict a corpus using a trained model, run the following command :

````shell
python ./run.py predict sequoia/sequoia-ud.parseme.frsemcor.simple.test ./models/sample_model.pkl  > models/predictions/sequoia-ud.parseme.frsemcor.simple.test.pred.conllu
````

Available arguments are showed below:

````
usage: run.py predict [-h] [--display] input_file model

positional arguments:
  input_file     path to input file
  model          path to trained model

options:
  -h, --help     show this help message and exit
  --display, -d  display the predictions
````

### Evaluation
Evaluate the predicted output with LAS/UAS :

````shell
python lib/accuracy.py --pred models/predictions/sequoia-ud.parseme.frsemcor.simple.test.pred.conllu --gold sequoia/sequoia-ud.parseme.frsemcor.simple.test --tagcolumn deprel 
````

---
### Semantic parsing

---

To use the parser for semantic parsing, simply add the ```--semantic``` argument before specifying the mode:

```
usage: run.py [-h] [--disable_cuda] [--semantic] {preprocess,train,predict} ...

Graph-based biaffine parser of French

positional arguments:
  {preprocess,train,predict}
                        Select mode: preprocess | train | predict
    preprocess          Preprocess data
    train               Train model
    predict             Predict semantic structures

options:
  -h, --help            show this help message and exit
  --disable_cuda        disable CUDA
  --semantic            enable semantic parser (default = syntactic)
```

To evaluate semantic prediction, launch the following command:

````shell
python lib/accuracy.py --pred models/predictions/sequoia-ud.parseme.frsemcor.simple.test.pred.conllu --gold sequoia/sequoia-ud.parseme.frsemcor.simple.test --tagcolumn deps 
````

**Note: Evaluation is performed only on the DEPS column. Make sure this column is correctly filled before running the evaluation.**

---

## Releases
- version 1.0.0 : Graph-based semantic parser using simple GRU and dynamic words dropout with similar configurations to which proposed by [^biaffine]. The model predicts head-governor dependencies only.
- version 1.1.0 : This graph-based semantic parser utilizes bidirectional LSTM/GRU layers with a word dropout rate of 0.33, maintaining configurations similar to the previous version. It has been extended to predict labeled dependencies for well-formed trees.
- version 1.2.0 : Pre-trained transformer is incorporate into our parser for prediction of dependency structures of well-formed trees.
- version 2.0.0 : Complete semantic parser and evaluation on DEPS column.

## Results
- version 1.0.0 achieved around 85% for UAS accuracy on the test corpus.
- version 1.1.0 achieved 86.96% for UAS and 84.98% for LAS on all head.
- version 1.2.0 with pre-trained camembert transformer achieved 81.56% for UAS and 75.85% for LAS on the test corpus.

## TO DO

---
- [X] Understanding the deep syntax structure of the Sequoia corpus, making it UD-compatible
- [X] Developing a biaffine classifier able to predict labeled syntactic trees
  - [X] Use a pre-trained transformer encoder instead of an RNN
- [X] Adapting the classifier to predict generic graphs instead of well-formed trees
- [X] Hyper-parameter optimisation of the system on the development corpus
- [X] Evaluation on the test portion of the deep syntax annotation of the Sequoia corpus
  - [X] In particular, implementing a script to evaluate the predictions

## References
[^biaffine]: Dozat et al. Stanford’s Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task. CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, pages 20–30, Vancouver, Canada, August 3-4, 2017. 