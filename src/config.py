SEQUOIA_SIMPLE_TRAIN = "sequoia/sequoia-ud.parseme.frsemcor.simple.train"
SEQUOIA_SIMPLE_TEST = "sequoia/sequoia-ud.parseme.frsemcor.simple.test"
SEQUOIA_SIMPLE_DEV = "sequoia/sequoia-ud.parseme.frsemcor.simple.dev"
TINY_CONLLU = "sequoia/tiny.conllu"

PAD_TOKEN = "PAD_ID"
PAD_TOKEN_VAL = 0
UNK_TOKEN = "UNK_ID"
UNK_TOKEN_VAL = 1

FORM_VOCAB = {PAD_TOKEN: PAD_TOKEN_VAL, UNK_TOKEN: UNK_TOKEN_VAL}
UPOS_VOCAB = {PAD_TOKEN: PAD_TOKEN_VAL, UNK_TOKEN: UNK_TOKEN_VAL}
DEPREL_VOCAB = {PAD_TOKEN: PAD_TOKEN_VAL, UNK_TOKEN: UNK_TOKEN_VAL}

UD_COLUMNS = ['form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel',
              'deps', 'misc', 'parseme:mwe', 'frsemcor:noun', 'parseme:ne']

CAMEMBERT_BASE = "almanach/camembert-base"
