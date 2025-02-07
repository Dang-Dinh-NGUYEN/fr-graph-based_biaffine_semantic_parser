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

### Data preprocessing

To prepare data from a given corpus, `./sequoia/src/sequoia-ud.parseme.frsemcor.simple.train` for example, and save
the output, run the following cell :

````shell
python ./src/run.py preprocess ./sequoia/src/sequoia-ud.parseme.frsemcor.simple.train -u -s
````

Data can also be preprocessed with existing vocabularies. The following shell presents the preprocessing on 
`./sequoia/src/sequoia-ud.parseme.frsemcor.simple.dev` with vocabularies extracted from the corpus `.train` : 

````shell
python ./src/run.py preprocess ./sequoia/src/sequoia-ud.parseme.frsemcor.simple.dev -l=./sequoia/src/sequoia-ud.parseme.frsemcor.simple.train -s
````

Other arguments are available as represented below :

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

## TO DO

---
- [ ] Understanding the deep syntax structure of the Sequoia corpus, making it UD-compatible
- [ ] Developing a biaffine classifier able to predict labeled syntactic trees
  - [ ] Use a pre-trained transformer encoder instead of an RNN
- [ ] Adapting the classifier to predict generic graphs instead of well-formed trees
- [ ] Hyper-parameter optimisation of the system on the development corpus
- [ ] Evaluation on the test portion of the deep syntax annotation of the Sequoia corpus
  - [ ] In particular, implementing a script to evaluate the predictions