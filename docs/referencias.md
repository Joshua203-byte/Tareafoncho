# Referencias del proyecto

Fuentes citables para el `REPORT.md`. Cada cifra indica dónde se verificó.

## Papers

| Tema | Cita | Enlace |
|---|---|---|
| Comparación de encoders y resultados en SQuAD 2.0 | He et al., 2021. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing* | https://arxiv.org/abs/2111.09543 |
| DeBERTa (atención desentrelazada) | He et al., 2020. *DeBERTa: Decoding-enhanced BERT with Disentangled Attention* | https://arxiv.org/abs/2006.03654 |
| DistilBERT (modelo elegido) | Sanh et al., 2019. *DistilBERT, a distilled version of BERT* | https://arxiv.org/abs/1910.01108 |
| BERT | Devlin et al., 2019. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | https://arxiv.org/abs/1810.04805 |
| RoBERTa | Liu et al., 2019. *RoBERTa: A Robustly Optimized BERT Pretraining Approach* | https://arxiv.org/abs/1907.11692 |
| ModernBERT | Warner et al., 2024. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder* | https://arxiv.org/abs/2412.13663 |
| SQuAD 2.0 (preguntas sin respuesta) | Rajpurkar et al., 2018. *Know What You Don't Know: Unanswerable Questions for SQuAD* | https://arxiv.org/abs/1806.03822 |
| SQuAD 1.0 | Rajpurkar et al., 2016. *SQuAD: 100,000+ Questions for Machine Comprehension of Text* | https://arxiv.org/abs/1606.05250 |
| Transformer (arquitectura base) | Vaswani et al., 2017. *Attention Is All You Need* | https://arxiv.org/abs/1706.03762 |

## Cifras de SQuAD 2.0 (dev set, F1 / EM)

Fuente: tabla de SQuAD 2.0 en el paper de DeBERTaV3 (He et al., 2021). Verificado vía búsqueda de la tabla, no leyendo el PDF completo. Antes de citarlo en el reporte, confirmar los números directamente en el paper.

| Modelo | F1 | EM |
|---|---|---|
| BERT-base | 76.3 | 73.7 |
| RoBERTa-base | 83.7 | 80.5 |
| DeBERTa-base | 86.2 | 83.1 |
| DeBERTa-v3-base | 88.4 | 85.4 |

No hay cifra verificada de DistilBERT ni de ModernBERT en SQuAD 2.0. Nuestra medición propia reemplazará esta tabla en la Fase 3, con el mismo harness para todos los candidatos.

## Conteos de parámetros y latencias

Medidos en este proyecto con `scripts/benchmark_models.py`; ver `logs/model_benchmark.json`.

## Datos

- SQuAD 2.0 (Hugging Face): https://huggingface.co/datasets/rajpurkar/squad_v2
- Estadísticas de exploración: `logs/data_stats.json`, generadas por `scripts/explore_data.py`.
