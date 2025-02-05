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
Before using this semantic parser, make sure that all necessary libraries and dependencies exist/installed by running the following command :

````shell
pip install -r requirements.txt
````

## Usage :

To prepare data from a given corpus, please run the following cell :
````shell
python ./src/run.py preprocess
````


Other arguments are available as represented below :

````
usage: run.py [-h] [--ftrain FTRAIN] [--fdev FDEV] [--ftest FTEST] [--save] [--load] [--preprocessed PREPROCESSED] [--display] {preprocess,train,predict}

Graph-based biaffine semantic parser of French

positional arguments:
  {preprocess,train,predict}

options:
  -h, --help            show this help message and exit

Data Options:
  --ftrain FTRAIN       path to train corpus
  --fdev FDEV           path to dev corpus
  --ftest FTEST         path to test corpus
  --save, -s            save preprocessed data
  --load, -l            load preprocessed data
  --preprocessed PREPROCESSED
                        path to preprocessed data
  --display, -d         display preprocessed data
````
## TO DO

---
- [ ] Understanding the deep syntax structure of the Sequoia corpus, making it UD-compatible
- [ ] Developing a biaffine classifier able to predict labeled syntactic trees
  - [ ] Use a pre-trained transformer encoder instead of an RNN
- [ ] Adapting the classifier to predict generic graphs instead of well-formed trees
- [ ] Hyper-parameter optimisation of the system on the development corpus
- [ ] Evaluation on the test portion of the deep syntax annotation of the Sequoia corpus
  - [ ] In particular, implementing a script to evaluate the predictions