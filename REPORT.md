# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `Fonchos` |
| Members | Pending: team member names |
| Repository URL | Pending |
| `submission` commit hash | Pending: filled in when the tag is created |
| Public service URL used at the dry run | Pending: tunnel from Phase 6 (ngrok chosen, not running yet) |
| AI tools used (which, and for what) | Claude Code (Claude Haiku 5.5 and Claude Sonnet 5.5) to write the service, the tests, the measurement scripts and this report. Every number comes from our own runs, which are cited in the repository. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert-base-uncased` (66.4 M parameters, 6 layers, hidden size 768). It is a base checkpoint with no SQuAD fine-tuning; we checked on the Hub that it does not carry the `squad` tag.
- **Alternatives considered:** `bert-base-uncased` (108.9 M parameters, 235 ms per question on CPU), `roberta-base` (124.1 M, 229 ms), `deberta-v3-base` (183.8 M, 387 ms) and `ModernBERT-base` (149.6 M, 317 ms), all measured in Phase 1 (`docs/fase1_datos_y_modelo.md`). We rejected them: with a 5 s timeout, the tunnel latency and a 150 s global cap for 50 items, a 12-layer encoder leaves little margin. We also rejected `distilbert-base-uncased-distilled-squad` because it is already trained on SQuAD, which the guide forbids.
- **Trade-off:** we give up some quality compared with the 12-layer models. In return, a short question takes 55–70 ms on CPU and the model weighs 265 MB.
- **Adaptation:** LoRA of rank 16 on the four attention matrices of every block, with the QA head trained in full. That is 591,362 trainable parameters (0.88 %). After training the adapters are merged, so the API loads a plain model without `peft`.

## 2. Training

| Item | Value |
|---|---|
| Training data | Full SQuAD 2.0 `train`: 130,319 questions (131,754 windows) |
| Validation data | `dev_select`: 5,917 questions from the SQuAD 2.0 `validation` split, divided by article. `dev_report` (5,956) was not used for any number in this report. |
| Epochs | Cap of 8 with early stopping (patience 2). The run stopped after epoch 5, and the service serves epoch 5. |
| Learning rate and schedule | 2e-4, linear schedule with 6 % warmup |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | RTX 4050 Laptop (6 GB). 6,529 s (≈ 1 h 49 min) up to epoch 5. |

- **Loss curve:** `logs/runs/r16_es/train_log.jsonl` (loss per step). There is no `logs/loss.png` image yet; we will generate it before submission.
- **Validation results (`dev_select`, epoch 5, τ = 0):** EM 66.17, F1 68.78, HasAns F1 64.72, NoAns accuracy 72.57. With the candidate τ from Section 3 (−2), the score on the guide's scale, reweighted to 25 % unanswerable, is 0.697. This is a proxy computed on the same set we used to choose, so it is inflated.

## 3. No-answer threshold

- **Scores compared:** the null score is `start[CLS] + end[CLS]`, taking the minimum across windows. The span score is the best valid start/end pair (top 20 per side, maximum length 30 tokens) across all windows. We answer with the span if `best_span − null > τ` (BERT §4.3 rule); otherwise we return `null`.
- **How we chose the value:** we swept τ on `dev_select` epoch by epoch (`scripts/sweep_tau.py`, `scripts/item_score.py`) and reweighted to 25 % unanswerable, the proportion expected in the hidden set. For epoch 5 the best τ is −2 (score 0.697), and with τ = −2 the score stays the same across the final epochs, so the value is stable.
- **Final value:** **pending**. The code defaults to `QA_TAU=0.0` until the team fixes the value once the final model is trained. It must be set before serving.
- **Too low or too high:** with a very low τ the model answers almost everything and accuracy on unanswerable questions drops. With a very high τ it returns `null` too often and loses valid answers (HasAns F1 falls).

## 4. Serving architecture

- **Framework and server:** FastAPI with uvicorn, **1 worker**. The model lives in a single process.
- **Model loading:** in a background thread at startup (*lifespan*). While it loads, `/health` answers 503 `loading`. If `QA_API_TOKEN` is missing, startup fails.
- **Request flow:** the token is checked first. Then the JSON is parsed and validated against the contract schema (length limits and extra fields). Next, `ValidationFeatures` splits the question and context into 384-token windows with a 128-token overlap, so long contexts are covered by several windows. One model pass over the batch of windows, on CPU and under a lock. `predict_example` applies τ and cuts the text out of the context using the offsets. An `assert` guarantees `context[start:start+len] == answer`.
- **Access control:** `Authorization: Bearer` is compared with `hmac.compare_digest`. Both the scheme and the token are case-sensitive. Without a valid token the answer is always 401, even if the body is broken. The value comes from `QA_API_TOKEN`, never from the code; `.env` is in `.gitignore`.
- **Tunnel:** ngrok (agent installed with winget). Pending: update the agent (the account requires 3.28.0 or newer), start it with `ngrok http 8000`, and test it from another network.
- **How to run** from a clean machine:

```bash
pip install -r requirements.txt
# Copy the models/ folder (not in git) or point QA_MODEL_DIR at it.
export QA_API_TOKEN=...            # PowerShell: $env:QA_API_TOKEN="..."
export QA_MODEL_DIR=models/r16_es/merged_epoch5
export QA_TAU=...                  # pending: set once the final model is trained
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `16` | 383 | yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | yes |
| sample-03 | `null` | `null` | null | yes |
| sample-04 | `null` | `3` | 310 | no |
| sample-05 | `50,257` | `50,257` | 3157 | yes |

sample-04 fails on a near miss: the context says "3 times" about GPT-3, but the question asks about BERT-base. The model matches the type of quantity and does not notice that the subject changed. Five items are not statistically significant; this figure is only a sanity check.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 (merged model; `/info` reports it) |
| Size on disk | 265 MB (`model.safetensors`, fp32) |
| Mean latency per `/predict` request on our laptop (CPU) | 55–70 ms for contexts of ≈ 450 characters; 466 ms for the 3,171-character context (several windows). Measured with `scripts/bench_predict.py` while another training session was running on the same machine. |
| HTTP smoke test (localhost, no tunnel) | 50 sequential requests in 7.3 s, slowest 488 ms, 0 failures (`scripts/smoke_test.py`) |
| Laptop CPU and RAM | Intel Core i7-12650H (10 cores, 16 threads), 16 GB RAM |

## 7. Problems encountered and how you fixed them

1. **Problem:** early stopping with τ = 0 was not reliable. HasAns F1 in the r16 run bounced between 53 and 70 from one epoch to the next. **Cause:** the τ = 0 threshold shifts with every epoch, so the score changes because of the threshold and not because of the model. **Fix:** select with the guide-scale score reweighted to 25 % unanswerable, and sweep τ per epoch. **Commit:** `9fb7163` (early stopping and selection) and `909035a` (per-item score).
2. **Problem:** `predict_example` returns a logit margin for spans but the raw null logit for `null`, while the contract requires a `score` between 0 and 1. **Cause:** the two branches use different scales. **Fix:** map it with a sigmoid in `service/qa.py`. It is still not a calibrated value. **Commit:** `498992e`.
3. **Problem:** in a worktree without `models/`, the tests that use the real model were silently skipped. **Cause:** the weights are not in git. **Fix:** the `QA_MODEL_DIR` variable and a visible reason in the `skip`. **Commit:** `0a391e2`.

## 8. Decoder investigation

Two problems of ours, measured on the public sample set, motivate the question:

- **sample-04:** the extractive model returns a span for an unanswerable question. A decoder trained to answer "no answer" in text could say so directly, but the evaluation requires an exact span match. A decoder that paraphrases breaks the rule `context[start:start+len] == answer`, so we would have to validate every output against the context.
- **Long contexts:** our 3,171-character context uses several windows. A decoder reading a long context would pay a higher attention cost on CPU.

The cost of a decoder on CPU and its real latency have not been measured yet. That is left for Phase 8.

## 9. Known limitations

- **Threshold:** τ is not fixed yet. The default `0.0` is not the calibrated value.
- **Score:** not calibrated. For `null` it is the null logit passed through a sigmoid, so it is not comparable with the score of a span.
- **Concurrency:** one worker and one lock. Requests queue up; with a 5,000-character context latency stays far below 5 s, but a long queue can exceed the limit.
- **Input:** English only. We do not check the language, and we have not measured text outside the SQuAD distribution.
- **Auth:** a single shared token, with no rotation or rate limiting.
- **Evaluation:** `dev_report` was not evaluated for this report. That single evaluation is decided by the team.
