"""Construcción del modelo: DistilBERT con cabeza de QA, con o sin LoRA.

Arquitectura
------------
DistilBERT es un *backbone* de 6 bloques Transformer más una *cabeza* de QA:

    embeddings -> 6 bloques (atención + feed-forward) -> qa_outputs (Linear 768 -> 2)

``qa_outputs`` produce un logit de inicio y uno de fin por token. Es la única
capa que no existe en el checkpoint preentrenado, por eso hay que entrenarla
siempre completa, incluso con LoRA.

Por qué LoRA solo en atención
-----------------------------
LoRA añade a cada matriz objetivo ``W`` una actualización de bajo rango
``(alpha / r) * B @ A``, donde solo ``A`` y ``B`` se entrenan y ``W`` queda
congelada. Aquí elegimos las cuatro matrices de atención de cada bloque
(``q_lin``, ``k_lin``, ``v_lin``, ``out_lin``), porque el paper de LoRA (Hu et al.,
2021, §7.1) apoya repartir un rango bajo entre varias matrices de atención. El
paper no adapta feed-forward ni embeddings, así que no los incluimos.

Merge
-----
Al terminar, ``merge_and_unload`` suma ``(alpha / r) * B @ A`` a ``W`` y elimina
los adaptadores. El modelo resultante tiene la misma arquitectura que el original
y el mismo número de parámetros, así que la API lo carga como un modelo normal.
"""

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForQuestionAnswering

ATTENTION_MODULES = ["q_lin", "k_lin", "v_lin", "out_lin"]


def build_model(model_id, lora_rank=0, lora_alpha=None, lora_dropout=0.1):
    """Carga el modelo de QA y, si ``lora_rank > 0``, le añade adaptadores LoRA.

    ``lora_rank = 0`` significa fine-tuning completo: todos los pesos entrenan.
    ``lora_alpha`` por defecto es ``2 * r``, una heurística común que deja el
    escalado en 2. Con ``alpha = r`` el escalado sería 1.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    if lora_rank == 0:
        return model
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_rank,
        lora_dropout=lora_dropout,
        target_modules=ATTENTION_MODULES,
        modules_to_save=["qa_outputs"],  # la cabeza de QA se entrena completa
        task_type=TaskType.QUESTION_ANS,
    )
    return get_peft_model(model, config)


def count_parameters(model):
    """Total de parámetros y cuántos se entrenan, para el reporte y el log."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": round(100 * trainable / total, 3)}


def merge_and_save(model, tokenizer, out_dir):
    """Fusiona los adaptadores LoRA (si los hay) y guarda un modelo normal en ``out_dir``.

    Tras el merge, ``out_dir`` contiene un modelo que se carga con
    ``AutoModelForQuestionAnswering.from_pretrained(out_dir)`` sin peft.
    """
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return model
