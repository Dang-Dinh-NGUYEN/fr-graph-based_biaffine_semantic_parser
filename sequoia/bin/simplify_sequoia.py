#!/usr/bin/env python3
"""
Simplify Sequoia corpus for pedagogical purposes:
- Remove all range tokens (e.g. "2-3 du" = "2 de" + "3 le"), keep only full tokens
  => Range tokens usually contain no annotation: they mark the presence of a contraction
  => The text may become strange to read, e.g. "L'ambassadrice de le Portugal à les Pays-Bas"
- Column FRSEMCOR:NOUN (TP4)
  - Remove all supersense annotations for multiword units 
    => keeping multiwords would make data preparation unnecessarily complex
  - Keep all simple (non-MWE) supersenses for NOUN, PROPN and NUM, remove others 
    => this allows the classifier to focus on these POS tags and ignore very low-frequency ones
  - For composed supersenses ("/", "x" or "+" operators), keep last element 
    - e.g. Artifact/Cognition -> Cognition
    => This reduces the tagset for prediction, making the task a bit easier for the classifier
- Column PARSEME:MWE
  - Keep in PARSEME:MWE only multiword expression annotations
  - Add a column PARSEME:NE with named entity annotations separately  
    => This allows working on named entity recognition as a standalone task (TP2)
  - Remove from PARSEME:NE:
    - (a) discontinuous NEs (e.g. [Jeanine] et Willy [Schaer]), 
    - (b) overlapping NEs (keep the longest one, choose randomly if same length)
  => Overlaps and discontinuities are hard to represent in sequence labelling models
- Columns HEAD, DEPREL and DEPS (TP5 and TP6)
  - Remove non-projective sentences
  => Non-projective parse trees are not straightforward to handle in the dependency parsing models we implement
  - [EXPERIMENTAL] Remove all deprel subrelations (after semicolon) to simplify the tagset
  => This simplification should be applied with caution. Although the number of deprel labels is relatively small,
  modifying them may lead to changes in the interpretation of dependency edges and potentially affect the
  syntactic/semantic structure. Exception on Deep Sequoia annotation.
  - Fulfill the DEPS columns
    - (a) For Sequoia-ud versions, which were aligned with the UD annotation:
        => The DEPS field of each token is filled with a label of the form head:deprel, where head and deprel are
        derived from the token's HEAD and DEPREL fields.
    - (b) For Sequoia.deep_and_surf versions:
        => The DEPS field of each token is filled with a suit of labels of the form head1:deprel1|head2:deprel2|...
        where head1 and head2 are the heads of the dependency, extracted from the token's HEAD field;
        and deprel1 and deprel2 are the label of each dependency, extracted from the token's DEPREL field
    => Evaluations should be taken on the DEPS column

This script depends on the `cuptlib` library. You can install it with:

git clone https://gitlab.com/parseme/cuptlib.git
cd cuptlib
pip install .
"""
import argparse
import sys
import conllu
import re
import pdb
import subprocess

try:
    import parseme.cupt as cupt
except ImportError:
    print("""Please install cuptlib before running this script\n\n  git clone \
  https://gitlab.com/parseme/cuptlib.git\n  cd cuptlib\n  pip install .""")
    sys.exit(-1)


#########################################

def remove_range_tokens(sent):
    range_counter = 0
    for (token_i, token) in enumerate(sent):
        if type(token["id"]) != int:  # Sentence ID is a complex object, remove it
            sent.pop(token_i)
            range_counter = range_counter + 1
    return range_counter


#########################################

def simplify_supersense(sent):
    del_ssense_counter = mod_ssense_counter = 0
    for token in sent:
        ssense_tags = token["frsemcor:noun"].split(";")
        for ssense_tag in ssense_tags:
            if ssense_tag[0].isdigit():  # Remove MWE supersense labels
                del_ssense_counter = del_ssense_counter + 1
                token["frsemcor:noun"] = "*"
            elif ssense_tag != "*" and token["upos"] not in ["NOUN", "PROPN", "NUM"]:
                del_ssense_counter = del_ssense_counter + 1
                token["frsemcor:noun"] = "*"
            elif "/" in ssense_tag or "+" in ssense_tag or "x" in ssense_tag:
                token["frsemcor:noun"] = re.split("[/+x]", ssense_tag)[-1]
                mod_ssense_counter = mod_ssense_counter + 1
            else:
                token["frsemcor:noun"] = ssense_tag
        if token["frsemcor:noun"] == "Felling":  # Correct typos in one SSense tag
            token["frsemcor:noun"] = "Feeling"
    return del_ssense_counter, mod_ssense_counter


#########################################

def simplify_mwe_ne(sent):
    ne_ind = 1  # Start new named entities at index 1 in new column
    del_ne_counter = 0
    mwes = cupt.retrieve_mwes(sent)  # get all MWE annotations
    ne_list = []
    mwe_list = []
    for mwe in mwes.values():
        if mwe.cat.startswith("PROPN|NE"):  # Named entity, add to NE list
            if mwe.n_gaps() == 0:
                ne_list.append(mwe)
            else:
                del_ne_counter = del_ne_counter + 1
        else:
            mwe_list.append(mwe)
    cupt.replace_mwes(sent, mwe_list)  # Clean all annotations, add only "MWE" ones

    def sorting_key(x):
        return (x.n_tokens(), len(sent) - sorted(list(x.span))[0], "final" in x.cat)

    ne_list_sort = sorted(ne_list, key=sorting_key, reverse=True)
    for token in sent:
        token["parseme:ne"] = "*"
    for ne in ne_list_sort:
        first_word = sorted(list(ne.span))[0] - 1  # -1 accesses list position, not token ID
        if sent[first_word]["parseme:ne"] == "*":  # No overlap, continue
            new_ne_cat = str(ne_ind) + ":" + ne.cat.split("-")[1].split(".")[0]
            sent[first_word]["parseme:ne"] = new_ne_cat
            for ne_i in sorted(list(ne.span))[1:]:
                sent[ne_i - 1]["parseme:ne"] = str(ne_ind)
            ne_ind = ne_ind + 1
        else:
            del_ne_counter = del_ne_counter + 1
    return del_ne_counter


#########################################

def is_projective(sent):
    for token in sent:
        dep_id = token["id"]
        head_ids = [int(h) for h in re.findall(r'\d+', token["deps"])]

        for head_id in head_ids:
            if head_id == dep_id:
                continue

            start, end = sorted((dep_id, head_id))

            for i in range(start + 1, end):  # exclusive
                token_i = sent[i - 1]  # 1-based to 0-based
                inner_heads = [int(h) for h in re.findall(r'\d+', token_i["deps"])]

                for inner_head in inner_heads:
                    if inner_head < start or inner_head > end:
                        return False
    return True


#########################################

def remove_subrelations(sent):
    subrel_counter = sum([1 if ':' in t['deprel'] else 0 for t in sent])
    for token in sent:
        token['deprel'] = re.sub(':.*', '', token['deprel'])
    return subrel_counter


def get_deep_syntax(heads: list, deprels: list):
    assert len(heads) == len(deprels), "Heads and deprels must be the same length"
    del_sr = mod_sr = 0  # Number of deleted/modified surface relations

    if len(heads) == 1:
        deprel = deprels[0]

        if deprel.startswith("S:"):
            # For 'S:' relations, keep only the first part after removing the prefix
            deprel = re.sub(r'^S:', '', deprel)
            deprel = re.split(r':', deprel)[0]
            mod_sr += 1
        else:
            # Otherwise keep the last part
            deprel = re.split(r':', deprel)[-1]

        deprels[0] = deprel

    else:
        new_heads = []
        new_deprels = []

        for head, deprel in zip(heads, deprels):
            if deprel.startswith('S:'):
                del_sr += 1
                continue  # Skip shallow dependencies

            # Strip 'D:' if present, then take the most specific (last) relation
            deprel = re.sub(r'^D:', '', deprel)
            deprel = re.split(r':', deprel)[-1]

            new_heads.append(head)
            new_deprels.append(deprel)

        heads = new_heads
        deprels = new_deprels

    return heads, deprels, del_sr, mod_sr


def fill_deps(sent, deep_syntax=False):
    del_sr_counter = mod_sr_counter = 0

    for token in sent:
        token_heads = token["head"].strip().split("|")
        token_deprels = token["deprel"].strip().split("|")

        if len(token_heads) != len(token_deprels):
            print(f"Warning {sent.metadata['sent_id']}: Unmatched sizes between heads {len(token_heads)}"
                  f" and deprels {len(token_deprels)}")

        if deep_syntax:  # Obtain deep syntax relations
            heads, deprels, del_sr, mod_sr = get_deep_syntax(token_heads.copy(), token_deprels.copy())
        else:
            heads = token_heads
            deprels = token_deprels
            del_sr = mod_sr = 0

        token_deps = "|".join([f"{head}:{deprel}" for head, deprel
                               in zip(heads, deprels)])
        token['deps'] = token_deps
        del_sr_counter = del_sr_counter + del_sr
        mod_sr_counter = mod_sr_counter + mod_sr
    return del_sr_counter, mod_sr_counter


#########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to preprocess and simplify the Sequoia corpus for academic use."

    )
    parser.add_argument('input_corpus', type=str, help='<input_corpus.conllu>')
    parser.add_argument('--deep_syntax', action='store_true',
                        help='obtain deep syntax relations (default = surface relations)')
    args = parser.parse_args()

    sequoia_field_parsers = {
        "head": lambda line, i: conllu.parser.parse_nullable_value(line[i]),
    }

    with open(args.input_corpus, "r", encoding="UTF-8") as f:
        np_counter = range_counter = del_ne_counter = 0
        del_ssense_counter = mod_ssense_counter = 0
        del_sr_counter = mod_sr_counter = 0
        np_ids = []

        for sent in conllu.parse_incr(f, field_parsers=sequoia_field_parsers):
            range_counter = range_counter + remove_range_tokens(sent)

            if not args.deep_syntax:
                del_ssense_ci, mod_ssense_ci = simplify_supersense(sent)
                del_ssense_counter = del_ssense_counter + del_ssense_ci
                mod_ssense_counter = mod_ssense_counter + mod_ssense_ci

                del_ne_counter = del_ne_counter + simplify_mwe_ne(sent)

                fill_deps(sent, args.deep_syntax)

                if is_projective(sent):  # Returns false to remove sentence
                    if sent.metadata.get("global.columns", None):  # Add header for new column
                        sent.metadata["global.columns"] += " PARSEME:NE"
                    print(sent.serialize(), end="")
                else:
                    np_counter += 1
                    np_ids.append(sent.metadata["sent_id"])
            else:
                del_sr_ci, mod_sr_ci = fill_deps(sent, args.deep_syntax)
                del_sr_counter = del_sr_counter + del_sr_ci
                mod_sr_counter = mod_sr_counter + mod_sr_ci

                print(sent.serialize(), end="")

    print("{} range tokens removed.\n".format(range_counter), file=sys.stderr)

    print("{} discontinuous and overlapping NEs removed.\n".format(del_ne_counter), file=sys.stderr)

    print("{} supersense tags removed (on MWEs or strange POS).".format(del_ssense_counter), file=sys.stderr)
    print("{} supersense tags modified (complex operators).\n".format(mod_ssense_counter), file=sys.stderr)

    print("{} surface relations removed .".format(del_sr_counter), file=sys.stderr)
    print("{} surface relations modified (surface-only relations).\n".format(mod_sr_counter), file=sys.stderr)

    print("{} non-projective sentences removed:".format(np_counter), file=sys.stderr)
    print(", ".join(np_ids), file=sys.stderr)
