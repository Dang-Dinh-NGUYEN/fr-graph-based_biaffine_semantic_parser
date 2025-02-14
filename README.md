# Graph-based biaffine semantic parser for french corpora

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
python ./run.py preprocess ./sequoia/sequoia-ud.parseme.frsemcor.simple.dev -l=./sequoia/sequoia-ud.parseme.frsemcor.simple.train -s
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
                        maximum sequence length (default=30)
  --save, -s            save preprocessed data
  --display, -d         display preprocessed data
````

### Training model
At the beginning, we define our model and train it with similar configurations/conditions as mentioned in the article of 
Dozat and Manning [^biaffine]. 

````shell
python ./run.py train -s --output=./models/sample_model.pkl
````
Other arguments are presented as follows :
````
usage: run.py train [-h] [--ftrain FTRAIN] [--fdev FDEV] [--ltrain LTRAIN] [--ldev LDEV] [--d_w D_W] [--d_t D_T] [--d_h D_H] [--d D] [--dropout DROPOUT] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--save]
                    [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --ftrain FTRAIN       path to train corpus
  --fdev FDEV           path to dev corpus
  --ltrain LTRAIN       path to preprocessed train file
  --ldev LDEV           path to preprocessed dev file
  --d_w D_W             dimension of word embeddings
  --d_t D_T             dimension of tag embeddings
  --d_h D_H             dimension of recurrent state
  --d D                 dimension of head/dependent vector
  --dropout DROPOUT     dropout rate
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
python ./run.py predict sequoia/sequoia-ud.parseme.frsemcor.simple.test ./sequoia-ud.parseme.frsemcor.simple.test.pred.conllu ./models/sample_model.pkl  
````

### Evaluation
Evaluate the predicted output with LAS/UAS :

````shell
python lib/accuracy.py --pred sequoia-ud.parseme.frsemcor.simple.test.pred.conllu --gold sequoia/sequoia-ud.parseme.frsemcor.simple.test --tagcolumn head 
````

## Version
- version 1.0.0 : graph-based semantic parser using simple GRU and dynamic word dropout with similar configurations to which proposed by [^biaffine]. The model predicts head-governor dependencies only


## TO DO

---
- [ ] Understanding the deep syntax structure of the Sequoia corpus, making it UD-compatible
- [ ] Developing a biaffine classifier able to predict labeled syntactic trees
  - [ ] Use a pre-trained transformer encoder instead of an RNN
- [ ] Adapting the classifier to predict generic graphs instead of well-formed trees
- [ ] Hyper-parameter optimisation of the system on the development corpus
- [ ] Evaluation on the test portion of the deep syntax annotation of the Sequoia corpus
  - [ ] In particular, implementing a script to evaluate the predictions

## References
[^biaffine] : Dozat et al. Stanford’s Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task. CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, pages 20–30, Vancouver, Canada, August 3-4, 2017. 