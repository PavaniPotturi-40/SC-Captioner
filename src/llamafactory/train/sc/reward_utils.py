'''
rewards.py: this file contains CAPTURE-related reward computation functions
'''

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import collections
import re
import difflib
import io
import contextlib
import numpy as np

# ✅ Replacement for missing encode_phrases
from sentence_transformers import SentenceTransformer

# Load model once globally
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_phrases(model_unused, list1, list2, batch_size=4):
    emb1 = _embedding_model.encode(list1, convert_to_numpy=True)
    emb2 = _embedding_model.encode(list2, convert_to_numpy=True)
    return emb1, emb2


# ---------- MERGE SCENE GRAPH ----------
def merge_sentence_results(results, text_processor):
    objects, attributes, relations = set(), collections.defaultdict(set), set()
    relations_original, attributes_original, objects_original, attributes_ = set(), collections.defaultdict(set), set(), set()

    for result in results:
        for entity in result['entities']:
            objects_original.add(entity['head'])
            lemmatized_obj = text_processor.normalize_word(entity['head'], wordnet.NOUN)
            objects.add(lemmatized_obj)

            for attribute in entity['attributes']:
                attributes_original[entity['head']].add(attribute)
                attributes_.add(attribute)
                attribute = text_processor.normalize_word(attribute, wordnet.ADJ)
                if ' of' in attribute:
                    continue
                attributes[lemmatized_obj].add(attribute)

        for relation in result['relations']:
            relations.add((
                text_processor.normalize_word(result['entities'][relation['subject']]['head'], wordnet.NOUN),
                relation['relation'],
                text_processor.normalize_word(result['entities'][relation['object']]['head'], wordnet.NOUN)
            ))

            relations_original.add((
                result['entities'][relation['subject']]['head'],
                relation['relation'],
                result['entities'][relation['object']]['head']
            ))

    return objects, attributes, relations, relations_original, attributes_original, objects_original, attributes_


# ---------- MAIN REVISION FUNCTION ----------
def get_revision(
    objects_1,
    objects_2,
    attributes_1,
    attributes_2,
    relations_1,
    relations_2,
    text_1,
    text_2,
    text_encoder,
    stop_words
):
    '''
    input: components of two sentences
    return: removed and added components
    '''

    # ---------- OBJECT DIFFERENCES ----------
    removed_objects = objects_1 - objects_2
    added_objects = objects_2 - objects_1

    removed_objects_cache = set()
    added_objects_cache = set()

    for obj in removed_objects:
        if obj not in text_2 and obj in text_1:
            removed_objects_cache.add(obj)

    for obj in added_objects:
        if obj not in text_1 and obj in text_2:
            added_objects_cache.add(obj)

    removed_objects = removed_objects_cache
    added_objects = added_objects_cache

    # ---------- SEMANTIC FILTER ----------
    removed_list = list(removed_objects)
    added_list = list(added_objects)

    if removed_list and added_list:
        with io.StringIO() as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                rem_emb, add_emb = encode_phrases(text_encoder, removed_list, added_list)

        sim_mat = rem_emb.dot(add_emb.T)

        remove_r, remove_a = [], []

        for i in range(len(removed_list)):
            for j in range(len(added_list)):
                if sim_mat[i, j] > 0.75:
                    remove_r.append(i)
                    remove_a.append(j)

        removed_list = [x for i, x in enumerate(removed_list) if i not in remove_r]
        added_list = [x for i, x in enumerate(added_list) if i not in remove_a]

    removed_objects = set(removed_list)
    added_objects = set(added_list)

    # ---------- RELATIONS ----------
    removed_relations = relations_1 - relations_2
    added_relations = relations_2 - relations_1

    # ---------- ATTRIBUTES ----------
    removed_attributes = collections.defaultdict(set)
    added_attributes = collections.defaultdict(set)

    all_keys = set(attributes_1.keys()) | set(attributes_2.keys())

    for key in all_keys:
        v1 = attributes_1.get(key, set())
        v2 = attributes_2.get(key, set())

        if v1 - v2:
            removed_attributes[key].update(v1 - v2)
        if v2 - v1:
            added_attributes[key].update(v2 - v1)

    # ---------- FILTER ATTRIBUTES ----------
    filtered_removed_attr = collections.defaultdict(set)
    filtered_added_attr = collections.defaultdict(set)

    for key in removed_attributes:
        if key in removed_objects:
            continue
        for attr in removed_attributes[key]:
            if attr not in text_2 and attr in text_1:
                filtered_removed_attr[key].add(attr)

    for key in added_attributes:
        if key in added_objects:
            continue
        for attr in added_attributes[key]:
            if attr not in text_1 and attr in text_2:
                filtered_added_attr[key].add(attr)

    removed_attributes = filtered_removed_attr
    added_attributes = filtered_added_attr

    # ---------- STOP WORD FILTER ----------
    if stop_words:
        stop_words_list = set()
        wnl = WordNetLemmatizer()

        filtered_removed = set()
        filtered_added = set()

        for word in removed_objects:
            if wnl.lemmatize(word, pos='n') not in stop_words_list:
                filtered_removed.add(word)

        for word in added_objects:
            if wnl.lemmatize(word, pos='n') not in stop_words_list:
                filtered_added.add(word)

        removed_objects = filtered_removed
        added_objects = filtered_added

    return removed_objects, added_objects, removed_relations, added_relations, removed_attributes, added_attributes
