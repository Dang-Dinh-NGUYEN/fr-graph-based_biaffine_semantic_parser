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

The project is mainly developed using python, pytorch, transformers, and conllu libraries.

## Project organisation

---

## Libraries and dependencies
Before using this semantic parser, make sure that all necessary libraries and dependencies are installed by running the following command :

````shell
pip install -r requirements.txt
````

## Usage :

### Data preprocessing

To prepare data from a given corpus, `./sequoia/src/sequoia-ud.parseme.frsemcor.simple.train` for example, and save
the output, run the following cell :

````shell
python ./run.py preprocess ./sequoia/sequoia-ud.parseme.frsemcor.simple.train -u -s
````

Data can also be preprocessed with existing vocabularies. The following shell performs the preprocessing on 
`./sequoia/sequoia-ud.parseme.frsemcor.simple.dev` with vocabularies extracted from the corpus `.train` : 

````shell
python ./run.py preprocess ./sequoia/sequoia-ud.parseme.frsemcor.simple.dev -l=./sequoia/preprocessed_sequoia-ud.parseme.frsemcor.simple.train.pt -s
````

Other arguments are available as presented below :

````
usage: run.py preprocess [-h] [--load LOAD] [--update] [--max_len MAX_LEN] [--save] [--display] input_file

positional arguments:
  input_file            path to input file

options:
  -h, --help            show this help message and exit
  --load, -l LOAD       path to preprocessed file
  --update, -u          update vocabulary during preprocessing
  --max_len, -m MAX_LEN
                        maximum sequence length (default=50)
  --save, -s            save preprocessed data
  --display, -d         display preprocessed data
````

### Training model
At the beginning, we define our model and train it with similar configurations/conditions as mentioned in the article of 
Dozat and Manning [^biaffine]. 

````shell
python ./run.py train --bidirectional -s --output=./models/sample_model.pkl
````
Other arguments are presented as follows :
````
usage: run.py train [-h] [--ftrain FTRAIN] [--fdev FDEV] [--ltrain LTRAIN] [--ldev LDEV] [--d_w D_W] [--d_t D_T] [--d_h D_H] [--d_arc D_ARC] [--d_rel D_REL] [--rnn_type {lstm,gru}] [--rnn_layers RNN_LAYERS] [--bidirectional]
                    [--dropout_rate DROPOUT_RATE] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--save] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --ftrain FTRAIN       path to train corpus
  --fdev FDEV           path to dev corpus
  --ltrain LTRAIN       path to preprocessed train file
  --ldev LDEV           path to preprocessed dev file
  --d_w D_W             dimension of form embeddings (default=100)
  --d_t D_T             dimension of upos embeddings (default=100)
  --d_h D_H             dimension of recurrent state (default=200)
  --d_arc D_ARC         dimension of head/dependent vector (default=400)
  --d_rel D_REL         dimension of deprel vector (default=100)
  --rnn_type, -t {lstm,gru}
                        type of rnn (default=lstm)
  --rnn_layers RNN_LAYERS
                        number of rnn's layer (default=3)
  --bidirectional       enable bidirectional
  --dropout_rate, -r DROPOUT_RATE
                        dropout rate (default=0.33)
  --n_epochs N_EPOCHS   number of training epochs
  --batch_size BATCH_SIZE
                        batch_size
  --lr LR               learning rate
  --save, -s            save trained model
  --output, -o OUTPUT   path to save model
````

### Predict
To predict a corpus using a trained model, run the following command :

````shell
python ./run.py predict sequoia/sequoia-ud.parseme.frsemcor.simple.test models/predictions/sequoia-ud.parseme.frsemcor.simple.test.pred.conllu ./models/sample_model.pkl  
````

Available arguments are showed below:

````
usage: run.py predict [-h] [--display] input_file output_file model

positional arguments:
  input_file     path to input file
  output_file    path to output file
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

## Releases
- version 1.0.0 : Graph-based semantic parser using simple GRU and dynamic words dropout with similar configurations to which proposed by [^biaffine]. The model predicts head-governor dependencies only
- version 1.1.0 : This graph-based semantic parser utilizes bidirectional LSTM/GRU layers with a word dropout rate of 0.33, maintaining configurations similar to the previous version. It has been extended to predict labeled dependencies for well-formed trees.

## Results
- version 1.0.0 achieved around 85% for UAS accuracy on test corpus
- version 1.1.0 achieved 86.96% for UAS and 84.98 for LAS on all head
## TO DO

---
- [X] Understanding the deep syntax structure of the Sequoia corpus, making it UD-compatible
- [X] Developing a biaffine classifier able to predict labeled syntactic trees
  - [X] Use a pre-trained transformer encoder instead of an RNN
- [ ] Adapting the classifier to predict generic graphs instead of well-formed trees
- [ ] Hyper-parameter optimisation of the system on the development corpus
- [ ] Evaluation on the test portion of the deep syntax annotation of the Sequoia corpus
  - [ ] In particular, implementing a script to evaluate the predictions

## References
[^biaffine] : Dozat et al. Stanford’s Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task. CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, pages 20–30, Vancouver, Canada, August 3-4, 2017. 