# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `Fonchos` |
| Members | Diego Emiliano Romero Castillo, Joshua Gabriel Garcilazo Avendaño, Monica Daniela Alvarenga Mejia, Carlos Eduardo Moreno Diaz, Violeta Natalia Jacobo Bonilla |
| Repository URL | Pending |
| `submission` commit hash | Pending: filled in when the tag is created |
| Public service URL used at the dry run | Pending: filled in at the dry run (ngrok fixed domain, already working) |
| AI tools used (which, and for what) | Claude Code (Claude Haiku 5.5 and Claude Sonnet 5.5) to write the service, the tests, the measurement scripts and this report. Every number comes from our own runs, which are cited in the repository. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert-base-uncased` (66.4 M parameters, 6 layers, hidden size 768). It is a base checkpoint with no SQuAD fine-tuning; we checked on the Hub that it does not carry the `squad` tag.
- **Alternatives considered:** `bert-base-uncased` (108.9 M parameters, 235 ms per question on CPU), `roberta-base` (124.1 M, 229 ms), `deberta-v3-base` (183.8 M, 387 ms) and `ModernBERT-base` (149.6 M, 317 ms), all measured in Phase 1 (`docs/fase1_datos_y_modelo.md`). We rejected them: with a 5 s timeout, the tunnel latency and a 150 s global cap for 50 items, a 12-layer encoder leaves little margin. We also rejected `distilbert-base-uncased-distilled-squad` because it is already trained on SQuAD, which the guide forbids.
- **Trade-off:** we give up some quality compared with the 12-layer models. In return, a short question takes about 50 ms on CPU and the model weighs 265 MB.
- **Adaptation:** full fine-tuning: all 66,364,418 parameters are trained, so there is nothing to merge and the API loads a plain model without `peft`. We first tried LoRA (rank 16 on the four attention matrices, 591,362 trainable parameters, best `dev_select` proxy 0.697) and kept it as a control; full fine-tuning scored higher (0.721 at epoch 3), so that is the model we serve.

## 2. Training

| Item | Value |
|---|---|
| Training data | Full SQuAD 2.0 `train`: 130,319 questions (131,754 windows) |
| Validation data | `dev_select`: 5,917 questions from the SQuAD 2.0 `validation` split, divided by article. `dev_report` (5,956) was not used for any number in this report. |
| Epochs | 4 epochs in total (2 in `full_ft_2ep`, then resumed to epoch 4 in `full_ft_4ep`). We select epoch 3 by the guide-scale proxy on `dev_select`, and the service serves that checkpoint. |
| Learning rate and schedule | 3e-5, constant after a 6 % warmup |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | RTX 4050 Laptop (6 GB). 2,358 s for epochs 1–2 plus 2,345 s for epochs 3–4 (≈ 78 min in total). |

- **Loss curve:** `logs/runs/full_ft_2ep/` and `logs/runs/full_ft_4ep/` (`train_log.jsonl`, loss per step). There is no `logs/loss.png` image yet; we will generate it before submission.
- **Validation results:** on `dev_select` (τ = −2, epoch 3) the guide-scale score, reweighted to 25 % unanswerable, is 0.721; it was used to pick the epoch and τ, so it is optimistic. On `dev_report` (5,956 questions, evaluated **once**, with the model and τ already fixed; `docs/fase4_umbral_y_evaluacion.md`, `logs/report/full_ft_4ep_epoch_3_tau-2.json`) we get EM 62.56, F1 66.27, HasAns F1 74.20, NoAns accuracy 57.83 and a guide-scale proxy of 0.694 (0.647 counting only exact matches). The `dev_report` figure is the clean one.

## 3. No-answer threshold

- **Scores compared:** the null score is `start[CLS] + end[CLS]`, taking the minimum across windows. The span score is the best valid start/end pair (top 20 per side, maximum length 30 tokens) across all windows. We answer with the span if `best_span − null > τ` (BERT §4.3 rule); otherwise we return `null`.
- **How we chose the value:** we swept τ on `dev_select` for every checkpoint (`scripts/item_score.py`) and reweighted to 25 % unanswerable, the proportion expected in the hidden set, because `dev_select` itself has 48 % unanswerable questions. The best τ is −2 for LoRA epoch 5 and for full fine-tuning epochs 2 and 3, and the score stays within a plateau for τ between −4 and −0.5.
- **Final value:** **τ = −2**, with the full fine-tuning epoch 3 model. The service defaults to it and `QA_TAU` can override it.
- **Too low or too high:** with a very low τ the model answers almost everything and accuracy on unanswerable questions drops. With a very high τ it returns `null` too often and loses valid answers (HasAns F1 falls).

## 4. Serving architecture

- **Framework and server:** FastAPI with uvicorn, **1 worker**. The model lives in a single process.
- **Model loading:** in a background thread at startup (*lifespan*). While it loads, `/health` answers 503 `loading`. If `QA_API_TOKEN` is missing, startup fails.
- **Request flow:** the token is checked first. Then the JSON is parsed and validated against the contract schema (length limits and extra fields). Next, `ValidationFeatures` splits the question and context into 384-token windows with a 128-token overlap, so long contexts are covered by several windows. One model pass over the batch of windows, on CPU and under a lock. `predict_example` applies τ and cuts the text out of the context using the offsets. An `assert` guarantees `context[start:start+len] == answer`.
- **Access control:** `Authorization: Bearer` is compared with `hmac.compare_digest`. Both the scheme and the token are case-sensitive. Without a valid token the answer is always 401, even if the body is broken. The value comes from `QA_API_TOKEN`, never from the code; `.env` is in `.gitignore`.
- **Tunnel:** ngrok 3.39.11 (installed from the Microsoft Store) with the free account's fixed domain, started with `ngrok http 8000` while the service listens on `127.0.0.1:8000`. The five guide `curl` tests and a 50-request smoke test passed through the public URL (`logs/service/smoke_test_ngrok.txt`). Still pending: the same test from another network (mobile data) and a second tunnel as backup.
- **Tokenizer:** the weights folder has no tokenizer, so the service uses `QA_TOKENIZER_DIR` if set, then the model folder, then `distilbert-base-uncased` (fine-tuning does not change the vocabulary).
- **How to run** from a clean machine:

```bash
pip install -r requirements.txt
# Copy the models/ folder (not in git) or point QA_MODEL_DIR at it.
export QA_API_TOKEN=...            # PowerShell: $env:QA_API_TOKEN="..."
export QA_MODEL_DIR=models/full_ft_4ep/epoch_3
export QA_TAU=-2                   # default, shown here for clarity
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `1024 and 16` | 374 | no |
| sample-02 | `3.5 days` | `3.5 days` | 437 | yes |
| sample-03 | `null` | `null` | null | yes |
| sample-04 | `null` | `3` | 310 | no |
| sample-05 | `50,257` | `50,257` | 3157 | yes |

sample-01 asks how many attention heads BERT-large has, and the model returns the span "1024 and 16" (the hidden size and the heads together) instead of "16": the span boundary is wrong although the right number is inside it. sample-04 fails on a near miss: the context says "3 times" about GPT-3, but the question asks about BERT-base. The model matches the type of quantity and does not notice that the subject changed. With 3 of 5 correct this is below what the previous LoRA model got on the same items (4 of 5), We checked that both misses are model errors and not service bugs: running the same model without the service, one window at a time, with no padding and the same τ = −2, gives identical answers and offsets for all five items (sample-05 uses 3 windows, so padding is exercised). In sample-01 the model prefers to start the span at "1024" (start logit 7.16) over "16" (6.01), and its margin over the null score is 6.2, so no reasonable τ changes it; in sample-04 the margin is 1.74, so it would answer `null` only with τ > 1.74, and moving τ for one item would overfit. Five items are not statistically significant: we chose the model with `dev_select` and measured it on `dev_report`, so this table is only a sanity check.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 (merged model; `/info` reports it) |
| Size on disk | 265 MB (`model.safetensors`, fp32) |
| Mean latency per `/predict` request on our laptop (CPU) | 37–41 ms for contexts of ≈ 450 characters; 330 ms for the 3,171-character context (several windows). Measured with `scripts/bench_predict.py` (`logs/service/bench_predict.txt`); the timings move by about 20 % between runs. |
| HTTP smoke test through the ngrok tunnel (same laptop) | 50 sequential requests in 13.6 s, slowest 565 ms, 0 failures: the tunnel adds about 150 ms per request (`logs/service/smoke_test_ngrok.txt`) |
| HTTP smoke test (localhost, no tunnel) | 50 sequential requests in 5.9 s, slowest 400 ms, 0 failures (`scripts/smoke_test.py`, `logs/service/smoke_test_local.txt`) |
| Laptop CPU and RAM | Intel Core i7-12650H (10 cores, 16 threads), 16 GB RAM |

## 7. Problems encountered and how you fixed them

1. **Problem:** early stopping with τ = 0 was not reliable. HasAns F1 in the r16 run bounced between 53 and 70 from one epoch to the next. **Cause:** the τ = 0 threshold shifts with every epoch, so the score changes because of the threshold and not because of the model. **Fix:** select with the guide-scale score reweighted to 25 % unanswerable, and sweep τ per epoch. **Commit:** `9fb7163` (early stopping and selection) and `909035a` (per-item score).
2. **Problem:** `predict_example` returned a logit margin for spans but the raw null logit for `null`, while the contract requires a `score` between 0 and 1. A confident `null` got a score of 0.0056. **Cause:** the two branches use different scales. **Fix:** `predict_example` now also returns the margin `best_span − null`, and `service/qa.py` turns the distance to the threshold into a sigmoid, `sigmoid(margin − τ)` for a span and `sigmoid(τ − margin)` for `null`. **Commit:** `498992e` (first sigmoid mapping) and `c05d43c` (margin score).
3. **Problem:** in a worktree without `models/`, the tests that use the real model were silently skipped. **Cause:** the weights are not in git. **Fix:** the `QA_MODEL_DIR` variable and a visible reason in the `skip`. **Commit:** `0a391e2`.

## 8. Decoder investigation

Two problems of ours, measured on the public sample set, motivate the question:

- **sample-04:** the extractive model returns a span for an unanswerable question. A decoder trained to answer "no answer" in text could say so directly, but the evaluation requires an exact span match. A decoder that paraphrases breaks the rule `context[start:start+len] == answer`, so we would have to validate every output against the context.
- **Long contexts:** our 3,171-character context uses several windows. A decoder reading a long context would pay a higher attention cost on CPU.

The cost of a decoder on CPU and its real latency have not been measured yet. That is left for Phase 8.

## 9. Known limitations

- **Threshold:** τ = −2 was calibrated on SQuAD-style data reweighted to 25 % unanswerable. The hidden set uses other texts, so the real proportion of `null` answers may differ.
- **Score:** it is 0.5 when the decision sits right at τ and approaches 1 the further it is from τ, for spans and for `null` alike. It is not a calibrated probability, and it changes when τ changes.
- **Concurrency:** one worker and one lock. Requests queue up; with a 5,000-character context latency stays far below 5 s, but a long queue can exceed the limit.
- **Input:** English only. We do not check the language, and we have not measured text outside the SQuAD distribution.
- **Auth:** a single shared token, with no rotation or rate limiting.
- **Known wrong answers:** the public sample questions sample-01 ("How many attention heads does BERT-large have?" → `1024 and 16` instead of `16`) and sample-04 ("By how many times does LoRA reduce the GPU memory requirement when fine-tuning BERT-base?" → `3` instead of `null`). Both are model errors: the span boundary is wrong in the first, and the second is a near miss where the context talks about a different model.
- **Evaluation:** `dev_report` was used exactly once; any later change of model or τ would make that number no longer clean.
