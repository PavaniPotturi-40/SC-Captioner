import functools
import tabulate
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import collections
import torch
import tqdm
import contextlib
import io
from sentence_transformers import SentenceTransformer
import numpy as np

import spacy
nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# ---------- SIMPLE ENCODER ----------
def encode_phrases(model, list1, list2, batch_size=4):
    emb1 = model.encode(list1, convert_to_numpy=True)
    emb2 = model.encode(list2, convert_to_numpy=True)
    return emb1, emb2


# ---------- TEXT PROCESSOR ----------
class TextProcessor:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def normalize_word(self, word, pos):
        return self.wnl.lemmatize(word, pos=pos)


# ---------- MAIN CLASS ----------
class CAPTURE:
    def __init__(self):
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.2
        self.text_processor = TextProcessor()
        self.stop_words_list = set()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device).eval()

    # ---------- PARSER (REPLACED) ----------
    def sample_to_parse_results(self, sample):
        sample_index, text = sample

        doc = nlp(text)

        objects = set()
        attributes = collections.defaultdict(set)
        relations = set()

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                obj = token.lemma_.lower()
                objects.add(obj)

                for child in token.children:
                    if child.pos_ == "ADJ":
                        attributes[obj].add(child.lemma_.lower())

            if token.pos_ == "VERB":
                subject, obj = None, None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.lemma_.lower()
                    if child.dep_ in ["dobj", "pobj"]:
                        obj = child.lemma_.lower()

                if subject and obj:
                    relations.add((subject, token.lemma_.lower(), obj))

        return sample_index, list(objects), attributes, relations

    # ---------- MATCH ----------
    def compute_match(self, all_cand, all_gt):
        set1 = set(all_cand)
        set2 = set(all_gt)

        intersection = len(set1 & set2)
        precision = intersection / (len(set1) + 1e-6)
        recall = intersection / (len(set2) + 1e-6)

        return precision, recall

    # ---------- OBJECT SCORE ----------
    def compute_objects(self, gt_parsed, cand_parsed):
        gt_objects, _, _ = gt_parsed
        cand_objects, _, _ = cand_parsed

        p, r = self.compute_match(cand_objects, gt_objects)
        f1 = 2 * p * r / (p + r + 1e-6)

        return p, r, f1

    # ---------- ATTRIBUTE SCORE ----------
    def compute_attributes(self, gt_parsed, cand_parsed):
        _, gt_attr, _ = gt_parsed
        _, cand_attr, _ = cand_parsed

        gt_set = set([a for v in gt_attr.values() for a in v])
        cand_set = set([a for v in cand_attr.values() for a in v])

        p, r = self.compute_match(cand_set, gt_set)
        f1 = 2 * p * r / (p + r + 1e-6)

        return p, r, f1

    # ---------- SEMANTIC SCORE ----------
    def semantic_score(self, text1, text2):
        emb1 = self.text_encoder.encode(text1, convert_to_tensor=True)
        emb2 = self.text_encoder.encode(text2, convert_to_tensor=True)
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

    # ---------- MAIN SCORE ----------
    def compute_score(self, gts, preds):
        object_scores = []
        attr_scores = []
        sem_scores = []

        for key in gts:
            gt = gts[key][0]
            pred = preds[key][0]

            gt_parsed = self.sample_to_parse_results((key, gt))[1:]
            pred_parsed = self.sample_to_parse_results((key, pred))[1:]

            _, _, obj_f1 = self.compute_objects(gt_parsed, pred_parsed)
            _, _, attr_f1 = self.compute_attributes(gt_parsed, pred_parsed)
            sem = self.semantic_score(gt, pred)

            object_scores.append(obj_f1)
            attr_scores.append(attr_f1)
            sem_scores.append(sem)

        final_score = (
            0.4 * np.mean(object_scores) +
            0.2 * np.mean(attr_scores) +
            0.4 * np.mean(sem_scores)
        )

        return {
            "object_f1": np.mean(object_scores),
            "attribute_f1": np.mean(attr_scores),
            "semantic": np.mean(sem_scores),
            "final_score": final_score
        }


# ---------- TEST ----------
if __name__ == "__main__":
    refs = {
        'example_0': [
            "A red car and a white truck are driving on a busy city street with trees"
        ],
    }

    preds = {
        'example_0': [
            "A car and truck moving on the road near trees"
        ]
    }

    evaluator = CAPTURE()
    score = evaluator.compute_score(refs, preds)

    print("\nRESULT:")
    for k, v in score.items():
        print(f"{k}: {v:.4f}")
