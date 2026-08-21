#!/usr/bin/env python3
"""PIGuard (leolee99/PIGuard), ACL 2025.

    python3 -m proba.run --detector piguard

**This adapter enables `trust_remote_code`.** The checkpoint ships a custom model class and will
not load without executing code from its repository, so measuring it at all means running vendor
code from the pinned revision below. That is a real choice, not a default: it is stated here, in
the file that makes it, and the revision is pinned so what executes is fixed rather than whatever
the hub branch holds today. A reader who will not run third-party code should skip this row.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("piguard", version="acl2025")
class PIGuard(_HFClassifier):
    display = "PIGuard (ACL 2025)"
    model_id = "leolee99/PIGuard"
    revision = "dd78b24e330193a22d2293ac66922dd4f982f563"
    positive_label = "INJECTION"
    trust_remote_code = True
