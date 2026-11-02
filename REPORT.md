# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `Fonchos` |
| Members | Diego Emiliano Romero Castillo, Joshua Gabriel Garcilazo Avendaño, Monica Daniela Alvarenga Mejia, Carlos Eduardo Moreno Diaz, Violeta Natalia Jacobo Bonilla |
| Repository URL | https://github.com/Joshua203-byte/Tareafoncho |
| `submission` commit hash | The commit tagged `submission`. This report is inside that commit, so its hash cannot be written here: `git rev-parse submission`. |
| Public service URL used at the dry run | Sent to the instructor before the slot (ngrok fixed domain); not written in the repository. |
| AI tools used (which, and for what) | Claude Code (Claude Haiku 5.5 and Sonnet 5.5): training code, API, tests, measurement scripts, plots and this report. Every number comes from our own runs, cited by file. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert-base-uncased` ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108); 66.4 M parameters, 6 layers, hidden size 768), with no SQuAD fine-tuning (we checked the Hub).
- **Alternatives considered:** `bert-base-uncased` (108.9 M parameters, 235 ms per question on CPU), `roberta-base` (124.1 M, 229 ms), `deberta-v3-base` (183.8 M, 387 ms), `ModernBERT-base` (149.6 M, 317 ms), against 110 ms for DistilBERT in the same benchmark (`logs/model_benchmark.json`). With a 5 s timeout, tunnel latency and a 150 s cap for 50 items, a 12-layer encoder leaves little margin for a modest quality gain.
- **Trade-off:** a lower quality ceiling in exchange for about 40 ms per short question on CPU and a 265 MB model.
- **Adaptation:** full fine-tuning (all 66,364,418 parameters). LoRA (rank 16) was our control and scored 0.697 on the selection metric against 0.721.

## 2. Training

| Item | Value |
|---|---|
| Training data | Full SQuAD 2.0 `train`: 130,319 questions (131,754 windows) |
| Validation data | `dev_select`: 5,917 questions of the SQuAD 2.0 `validation` split, divided by article. `dev_report`: the other 5,956, evaluated once. |
| Epochs | 4 trained (2 in `full_ft_2ep`, resumed to 4 in `full_ft_4ep`); epoch 3 selected and served |
| Learning rate and schedule | 3e-5, constant after 6 % warmup |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | RTX 4050 Laptop (6 GB); 2,358 s + 2,345 s (≈ 78 min) |

![Training loss](logs/loss.png)

- **Where the hyperparameters come from:** the source papers, not a search of our own (no learning-rate sweep was run). Learning rate 3e-5, batch size 32 and 2–4 epochs lie inside the fine-tuning grid of [BERT](https://arxiv.org/abs/1810.04805) (App. A.3: lr {2e-5, 3e-5, 5e-5}, batch {16, 32}, epochs {2, 3, 4}). The 384/128 window is the setting of the [Hugging Face QA course](https://huggingface.co/learn/llm-course/chapter7/7). The 6 % warmup is a heuristic: BERT gives none for fine-tuning and [LoRA](https://arxiv.org/abs/2106.09685) uses 0.06. The LoRA control follows its §7.1 (low rank spread over the four attention matrices), and its lr 2e-4 is extrapolated from LoRA's GLUE settings. The epoch was chosen on validation, not taken from a paper.
- **Loss curve:** `logs/loss.png`, drawn by `scripts/plot_training.py` from `logs/runs/*/train_log.jsonl`. The loss falls at each epoch boundary, a sign that the model starts to memorise the training set, so we pick the epoch by validation, not by loss.
- **Validation results:**

| Set | EM | F1 | F1 answerable | Accuracy unanswerable | Guide-scale score |
|---|---|---|---|---|---|
| `dev_select`, epoch 3 (used to choose epoch and τ) | 64.41 | 68.02 | 77.91 | 58.78 | 0.721 |
| `dev_report`, evaluated once (clean) | 62.56 | 66.27 | 74.20 | 57.83 | 0.694 |

Epoch 4 has higher EM and F1 but a lower guide-scale score (0.709): it handles unanswerable questions better and answerable ones worse, and the guide scores both (`logs/dev_select.png`).

## 3. No-answer threshold

- **Decision rule:** the null score is `start[CLS] + end[CLS]` (minimum over windows); the span score is the best valid start/end pair over all windows. We answer with the span if `best span − null > τ`, otherwise `null` ([BERT](https://arxiv.org/abs/1810.04805) §4.3).
- **Choosing τ:** we swept τ on `dev_select`, reweighted to 25 % unanswerable (the hidden-set share; `dev_select` has 48 %). The best value is **τ = −2**, on a plateau between −4 and −0.5, and it is the service default (`QA_TAU`).

![Threshold sweep](logs/tau_sweep.png)

- **Too low or too high:** τ = −∞ always answers (score 0.630); τ = +∞ always says `null` and falls to the 0.25 baseline; τ = 4 answers only 34 % of questions (0.585). The threshold alone is worth about 9 points (0.630 → 0.721).

## 4. Serving architecture

- **Framework:** FastAPI with uvicorn, **1 worker**.
- **Model loading:** in a background thread at startup; `/health` answers 503 `loading` until the model is ready.
- **Request flow:** token check → JSON and schema validation → question and context split into 384-token windows with 128 overlap (a 5,000-character context uses several) → one batched forward pass on CPU under a lock → τ decides span or `null` → the text is cut from the original context with character offsets, and an assertion guarantees `context[start:start+len] == answer`.
- **Access control:** `Authorization: Bearer`, compared with `hmac.compare_digest` before the body is read, so a bad token is always 401. The token comes from `QA_API_TOKEN`, never from code; the service will not start without it, and `.env` is git-ignored.
- **Tunnel:** ngrok (free fixed domain) as main, Cloudflare Quick Tunnel as backup, both opened with `scripts/tunnel.ps1`; main command `ngrok http 8000`. Both pass the five guide `curl` tests, a 50-request smoke test and a 42-case robustness battery (`logs/service/`, `docs/fase6_tunel.md`).
- **How to run** from a clean machine:

```bash
pip install -r requirements.txt
# copy models/ (weights are not in git) and put the token in .env
export QA_API_TOKEN=...                      # PowerShell: $env:QA_API_TOKEN="..."
export QA_MODEL_DIR=models/full_ft_4ep/epoch_3
export QA_TOKENIZER_DIR=models/full_ft_4ep/merged
uvicorn service.app:app --host 127.0.0.1 --port 8000 --workers 1   # or .\scripts\run.ps1
ngrok http 8000                              # or .\scripts\tunnel.ps1
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `1024 and 16` | 374 | no |
| sample-02 | `3.5 days` | `3.5 days` | 437 | yes |
| sample-03 | `null` | `null` | null | yes |
| sample-04 | `null` | `3` | 310 | no |
| sample-05 | `50,257` | `50,257` | 3157 | yes |

Both errors are the model's, not the service's: the same model run outside the service, one window at a time without padding, gives identical answers and offsets on these five items and on 1,000 `dev_select` questions (`logs/service/eval_service_sample.txt`). In sample-01 the model starts the span at "1024" (start logit 7.16) instead of "16" (6.01), a margin of 6.2 over the null score that no reasonable τ changes. sample-04 is a near miss: the context gives "3 times" for GPT-3, the question asks about BERT-base; its margin is 1.74, so only τ > 1.74 returns `null`, and tuning τ for one item would overfit.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 265 MB (`model.safetensors`, fp32) |
| Mean latency per `/predict` request on our laptop (CPU) | 37–41 ms for ≈ 450-character contexts; 330 ms for the 3,171-character context (3 windows) (`logs/service/bench_predict.txt`). Over 1,000 `dev_select` questions: mean 71 ms, p95 112 ms. |
| Through the tunnel (50 sequential requests) | ngrok 13.6 s, Cloudflare 13.9 s, slowest 608 ms; local 5.9 s. The tunnel adds about 150 ms per request. |
| Laptop CPU and RAM | Intel Core i7-12650H (10 cores, 16 threads), 16 GB RAM |

## 7. Problems encountered and how you fixed them

1. **Problem:** early stopping with τ = 0 picked a bad epoch (answerable F1 swung between 53 and 70). **Cause:** τ = 0 moves every epoch, so the metric tracks the threshold, not the model. **Fix:** select by the guide-scale score at 25 % unanswerable, with τ swept per epoch. **Commit:** `9fb7163`, `909035a`.
2. **Problem:** a confident `null` got `score` 0.0056, but the contract needs [0, 1]. **Cause:** `predict_example` returned a margin for spans and the raw null logit for `null`, two scales. **Fix:** a sigmoid of the distance to τ. **Commit:** `498992e`, `c05d43c`.
3. **Problem:** real-model tests were skipped silently without weights. **Cause:** weights are not in git. **Fix:** `QA_MODEL_DIR`, a visible skip reason and a service-versus-reference test. **Commit:** `0a391e2`, `832b405`.
4. **Problem:** ngrok refused the agent installed by winget (`ERR_NGROK_121`). **Cause:** 3.3.1 is below the 3.28.0 minimum. **Fix:** install from the Microsoft Store (3.39.11). **Commit:** `53db785`.

## 8. Decoder investigation

A decoder-only model (GPT-1: 117 M parameters, 512-token context; [Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)) or an encoder–decoder (e.g. T5, [Raffel et al., 2020](https://arxiv.org/abs/1910.10683)) generates the answer as text instead of picking a span. We tie it to three problems measured on our model; the cost estimates are orders of magnitude, not measurements.

**1. Near-miss unanswerable questions (sample-04; only 57.8 % accuracy on unanswerable questions in `dev_report`, our weakest result).** SQuAD 2.0 unanswerable questions are written to resemble answerable ones ([Rajpurkar et al., 2018](https://arxiv.org/abs/1806.03822)). We compare two scalar scores at `[CLS]`; a generative model reads the whole question and can output "unanswerable", so it could notice that BERT-base is not GPT-3. *Cost:* 1.8× the parameters (GPT-1) to 3.3× (T5-base, 220 M); full fine-tuning of larger models does not fit in 6 GB of VRAM, so LoRA, which lost 2.4 points in our control; and we lose τ (worth 9 points) and must rebuild it from sequence probabilities.

**2. A wrong span boundary (sample-01, "1024 and 16").** Start and end come from independent logits, so an incoherent pair can win; a decoder conditions each token on the previous ones. *Cost:* the contract needs `context[start:start+len] == answer`, but generation can paraphrase ("16 heads"), so we need constrained decoding or a fallback to `null`, and `start` must be found by text search, which is ambiguous if the string repeats.

**3. Long contexts (3 windows: 330 ms against about 40 ms for one).** Windows need a merge rule; a longer-context decoder reads the paragraph once. *Cost:* GPT-1's 512 tokens cannot hold sample-05 (665 tokens) either, and GPT-2's 1,024 ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) is borderline for 5,000 characters. Compute per token is about 2× ours (85 M against 43 M non-embedding parameters) plus one pass per generated token: about 0.5 s for a short answer, but models with long context have billions of parameters, against a 5 s limit on a laptop CPU.

References: [Radford et al. (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), GPT-1; [Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), GPT-2; [Raffel et al. (2020)](https://arxiv.org/abs/1910.10683), T5; [Rajpurkar et al. (2018)](https://arxiv.org/abs/1806.03822), SQuAD 2.0; [Devlin et al. (2019)](https://arxiv.org/abs/1810.04805), BERT §4.3 and A.3; [Hu et al. (2021)](https://arxiv.org/abs/2106.09685), LoRA §7.1; [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108), DistilBERT.

## 9. Known limitations

- **Wrong answers:** sample-01 and sample-04 (Section 5), both model errors; with 50 hidden questions the score varies by about ±0.06 by chance alone.
- **Distribution shift:** τ and all scores come from SQuAD-style text; the hidden set uses other texts, so its `null` share and score may differ.
- **Score:** not a calibrated probability; it moves if τ moves.
- **Concurrency, input, auth:** one worker and one lock, so simultaneous requests queue; English only; one shared token, no rotation, no rate limit.
- **Availability:** the service needs this laptop on, plugged in and online with the tunnel running; the Cloudflare backup was tested from the same laptop only.
