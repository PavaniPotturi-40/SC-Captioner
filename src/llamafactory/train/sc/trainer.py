# ----------- IMPORTS -----------
import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union, List, Any

import torch
import torch.nn.functional as F
from transformers import Trainer, GenerationConfig
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

import torch.nn as nn

# ✅ FIXED IMPORTS
from .reward_utils import *
from nltk.corpus import stopwords
stop_words_list = set(stopwords.words('english'))

from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

import collections
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


# ----------- TRAINER -----------
class CustomSCTrainer(OnlineDPOTrainer):
    def __init__(
        self,
        model,
        ref_model,
        finetuning_args,
        processor,
        model_type=None,
        disable_dropout=True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.model_type = model_type
        self.ref_model = ref_model

        self.stop_words_list = set(stop_words_list)

        # ✅ text encoder (for similarity)
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # DPO params
        self._beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss

        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.9,
            do_sample=True,
        )

        Trainer.__init__(self, model=model, **kwargs)

    # ----------- SIMPLE PARSER (REPLACEMENT) -----------
    def simple_parse(self, text):
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
                subj, obj = None, None
                for child in token.children:
                    if child.dep_ in ["nsubj", "dobj", "pobj"]:
                        if subj is None:
                            subj = child.lemma_.lower()
                        else:
                            obj = child.lemma_.lower()

                if subj and obj:
                    relations.add((subj, token.lemma_.lower(), obj))

        return objects, attributes, relations

    # ----------- REWARD FUNCTION -----------
    def compute_rewards(self, first_texts, second_texts, prompts):
        rewards = torch.zeros(len(second_texts))

        for i in range(len(second_texts)):
            text_ref = second_texts[i]
            text_rej = first_texts[i]
            text_gt = prompts["chosen_text"][i]

            obj_ref, attr_ref, rel_ref = self.simple_parse(text_ref)
            obj_rej, attr_rej, rel_rej = self.simple_parse(text_rej)
            obj_gt, attr_gt, rel_gt = self.simple_parse(text_gt)

            # differences
            removed = obj_rej - obj_ref
            added = obj_ref - obj_rej

            # simple reward
            reward = len(added & obj_gt) - len(removed & obj_gt)

            rewards[i] = reward

        return rewards

    # ----------- LOSS -----------
    def _compute_stage2_loss(self, model, ref_model, first, second, prompts):
        first_logits = model(**first).logits
        second_logits = model(**second).logits

        first_logprobs = F.log_softmax(first_logits, dim=-1)
        second_logprobs = F.log_softmax(second_logits, dim=-1)

        reward = self.compute_rewards(
            prompts["first_response"],
            prompts["second_response"],
            prompts
        )

        loss = -(second_logprobs.mean() * reward.mean())

        return loss

    # ----------- TRAIN STEP -----------
    def training_step(self, model, inputs):
        model.train()

        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
            "chosen_text": inputs["chosen_text"],
        }

        # fake responses (simplified)
        first = {"input_ids": inputs["first_completion_input_ids"]}
        second = {"input_ids": inputs["completion_input_ids"]}

        prompts["first_response"] = ["dummy"] * len(inputs["chosen_text"])
        prompts["second_response"] = ["dummy"] * len(inputs["chosen_text"])

        loss = self._compute_stage2_loss(model, self.ref_model, first, second, prompts)

        self.accelerator.backward(loss)

        return loss.detach()
