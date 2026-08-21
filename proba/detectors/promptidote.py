#!/usr/bin/env python3
"""promptidote — a hosted prompt-injection detector, reached over its HTTP API.

    export PROMPTIDOTE_KEY=...            # or AIG_KEY
    python3 -m proba.run --detector promptidote

**THE TEXT LEAVES THE MACHINE.** Every document is sent to a third party's service. The carriers
in this corpus are scrubbed of contacts before publication, which is why this is tolerable here at
all — but it is still a transfer, and it is stated in the adapter that performs it rather than
left for a reader to deduce from the endpoint. Anyone measuring a detector of their own over
someone else's API should decide that separately from deciding to run the harness.

APERTURE. The service accepts 5000 characters per request, so that is what this adapter declares;
the base class splits longer documents on sentence boundaries and folds the windows back by max.
`max` is the aggregation the product itself uses (`any(window verdict)`), so a document is as
injected as its most injected window — taking the last window's score instead, which is what a
naive read of the response stream gives, would be wrong precisely on the long documents.

A FAILED REQUEST STOPS THE RUN. It does not score 0.0. An adapter that swallows service errors
produces a complete, plausible result that reads "the detector found nothing", and nothing about
the number looks wrong afterwards. Stopping is safe here because the runner checkpoints every
batch: a re-invocation resumes from the last one rather than re-paying for the whole pass.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..detector import Detector, register

ENDPOINT = os.environ.get("PROMPTIDOTE_ENDPOINT", "https://promptidote.tech/api/v1/check")
#: what the service accepts per request
API_MAX_CHARS = 5000
RETRIES = 4


@register("promptidote", version="api-v1")
class Promptidote(Detector):
    display = "promptidote"
    max_chars = API_MAX_CHARS
    notes = f"hosted API · {ENDPOINT} · window {API_MAX_CHARS} chars, folded by max"

    def setup(self):
        self.key = os.environ.get("PROMPTIDOTE_KEY") or os.environ.get("AIG_KEY")
        if not self.key:
            raise SystemExit(
                "promptidote needs an API key: set PROMPTIDOTE_KEY (or AIG_KEY).\n"
                "  The key is read from the environment and never written into a result file.")
        self.calls = 0

    def _post(self, text: str) -> float:
        body = json.dumps({"text": text}).encode("utf-8")
        last = None
        for attempt in range(RETRIES):
            req = urllib.request.Request(
                ENDPOINT, data=body, method="POST",
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                score = payload.get("score")
                if not isinstance(score, (int, float)):
                    raise ValueError(f"no numeric `score` in the response: {payload!r:.200}")
                self.calls += 1
                return float(score)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                    json.JSONDecodeError, TimeoutError) as exc:
                last = exc
                # A 4xx is a contract problem and will not fix itself; only back off on the rest.
                if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 \
                        and exc.code != 429:
                    break
                time.sleep(2 ** attempt)
        raise SystemExit(
            f"[promptidote] the service failed after {RETRIES} attempts: {last!r}\n"
            f"  The run stops here rather than scoring 0.0 and filing a result that reads as "
            f"'found nothing'.\n"
            f"  Scores already computed are checkpointed — re-run the same command to resume.")

    def score(self, docs):
        base = getattr(self, "progress_offset", 0)
        total = getattr(self, "progress_total", 0)
        t0 = time.time()
        for n, d in enumerate(docs, 1):
            yield self._post(d.text)
            if total and n % 200 == 0:
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  promptidote · documents {base}/{total} · {n} windows in batch, "
                      f"{rate:.1f} windows/s", flush=True)
