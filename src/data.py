"""Carga de datos para entrenamiento y evaluación.

Dos formas de preparar los datos, según el uso:

- **Entrenamiento** (``train_dataset``): un ``datasets.Dataset`` con las columnas
  que consume el ``Trainer``. Las etiquetas de inicio y fin ya van calculadas.
  Se hace con ``datasets.map``, que guarda el resultado en caché.
- **Evaluación** (``eval_features``): listas de Python con los offsets de cada
  ventana. Se usan en el post-procesamiento, que necesita los offsets, así que
  no pasan por ``Trainer``.
"""

from datasets import load_dataset

from src.features import TrainFeatures, ValidationFeatures

SEED = 42


def load_squad():
    """SQuAD 2.0 desde Hugging Face. Devuelve un ``DatasetDict`` con ``train`` y ``validation``."""
    return load_dataset("rajpurkar/squad_v2")


def train_dataset(tokenizer, raw_train, n_examples=None, max_length=384, stride=128, seed=SEED):
    """Dataset de entrenamiento tokenizado en ventanas.

    ``n_examples`` toma un subconjunto aleatorio reproducible (para el piloto).
    ``None`` usa el train completo.
    """
    if n_examples is not None:
        raw_train = raw_train.shuffle(seed=seed).select(range(n_examples))
    features = TrainFeatures(tokenizer, max_length=max_length, stride=stride)
    return raw_train.map(features, batched=True, remove_columns=raw_train.column_names)


def eval_features(tokenizer, raw_examples, max_length=384, stride=128):
    """Ventanas de evaluación con offsets, como diccionario de listas.

    ``raw_examples`` es un ``Dataset`` o una vista de uno. Devuelve un dict con
    ``input_ids``, ``attention_mask``, ``example_id`` y ``offset_mapping``.
    """
    features = ValidationFeatures(tokenizer, max_length=max_length, stride=stride)
    return features(raw_examples[:])
