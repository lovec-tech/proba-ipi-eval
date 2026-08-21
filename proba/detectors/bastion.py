#!/usr/bin/env python3
"""Bastion binary prompt-protection, DeBERTa-v3-xsmall.

    python3 -m proba.run --detector bastion

Its positive class is called `ATTACK` rather than `INJECTION` — declared here so the base class
resolves the right logit instead of falling back to index 1 by luck.

Licence note: the checkpoint is AGPL-3.0. That constrains how it may be deployed; it does not
constrain measuring it, and the row is reported like any other.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("bastion", version="deberta-v3-xsmall-v1")
class Bastion(_HFClassifier):
    display = "Bastion Prompt Protection (DeBERTa v3 xsmall)"
    model_id = "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1"
    revision = "3a5bbe0e8eadf86213378e4806da42a1a3177df8"
    positive_label = "ATTACK"
