"""Preprocesamiento de QA extractivo: de ejemplos de SQuAD a ventanas tokenizadas.

Estructura
----------
``BaseQAFeatures`` contiene todo lo que entrenamiento y validación comparten:
la llamada al tokenizer con ventanas deslizantes. Las subclases solo cambian lo
que hacen con cada ventana (``build``):

- ``TrainFeatures``: calcula las etiquetas ``start_positions`` y ``end_positions``.
- ``ValidationFeatures``: guarda ``example_id`` y los offsets del contexto para
  reconstruir la respuesta después.

Así la tokenización vive en un solo lugar y no puede divergir entre ambos flujos.

Por qué herencia (y no funciones sueltas con un flag)
-----------------------------------------------------
Con dos variantes que comparten todo menos un paso, una base con un método
``build`` que cada subclase implementa es lo más legible: quien lee
``TrainFeatures`` ve de inmediato qué cambia respecto a la validación, sin
buscar ramas ``if train:`` dentro de una función larga.

Un flag booleano dentro de una sola función también evitaría la herencia, pero
mezcla dos responsabilidades y crece mal: cada nueva variante agregaría otro
``if``. Si algún día hubiera muchas variantes que cambian pasos distintos (por
ejemplo, distintas estrategias de etiquetado), convendría componer objetos
pequeños en lugar de heredar. Con dos variantes, eso sería sobre-ingeniería.

Lo que la herencia **no** debe permitir es que cada subclase reimplemente la
tokenización. Por eso ``tokenize`` y ``__call__`` viven en la base y las
subclases solo sobrescriben ``build``.

Padding dinámico
----------------
Aquí NO hacemos padding. Cada ventana sale con su longitud real, y el padding
se hace por lote, justo antes de entrar al modelo, con ``make_collator``.

Por qué: la mediana de entrada es de ~158 tokens y el máximo es 384. Si
rellenáramos todo a 384, la mayoría del cómputo iría a tokens de relleno que
la atención igual procesa. Con padding por lote, cada lote se rellena solo
hasta la ventana más larga que contiene, así que el costo baja a casi el
costo de los tokens reales. Medido sobre el train completo: los tokens reales
son el 44 % de lo que procesaría un padding fijo a 384, es decir, ~2.25× menos
cómputo (ver logs/preprocessing_stats.json).

Costo: el collator debe recibir las etiquetas como enteros y las ventanas como
listas de longitud variable. Por eso ``remove_columns`` en el ``map`` debe quitar
``offset_mapping`` del entrenamiento: no es una entrada del modelo y el collator
no sabe qué hacer con listas de offsets.
"""

from transformers import DataCollatorWithPadding

CLS_INDEX = 0  # posición del [CLS]: en el entrenamiento significa "la respuesta no está en esta ventana"


def make_collator(tokenizer):
    """Devuelve el collator que hace padding dinámico por lote.

    Se usa en el DataLoader (o en ``Trainer(data_collator=...)``). Rellena
    ``input_ids`` y ``attention_mask`` hasta la ventana más larga del lote y
    deja ``start_positions`` y ``end_positions`` como enteros.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer)


class BaseQAFeatures:
    """Tokenización compartida de pregunta + contexto con ventanas deslizantes.

    Parámetros
    ----------
    tokenizer:
        Debe ser un tokenizer *fast* del mismo checkpoint que el modelo.
        Los offsets (``return_offsets_mapping``) solo existen en la versión fast.
    max_length:
        Tokens máximos por ventana, incluyendo pregunta y tokens especiales.
    stride:
        ``doc_stride``: tokens que se repiten entre ventanas consecutivas.
        Si una respuesta cae en la zona compartida, aparece completa en ambas.
    """

    def __init__(self, tokenizer, max_length=384, stride=128):
        if not tokenizer.is_fast:
            raise ValueError("Se necesita un tokenizer fast para return_offsets_mapping.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def tokenize(self, examples):
        """Tokeniza un lote de ejemplos en ventanas, sin padding.

        ``truncation="only_second"`` recorta solo el contexto. La pregunta
        tiene como máximo 500 caracteres, así que nunca supera la ventana.
        """
        return self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,  # una fila por ventana, no por ejemplo
            return_offsets_mapping=True,  # offset (inicio, fin) en caracteres de cada token
        )

    def __call__(self, examples):
        """Función para ``datasets.map(..., batched=True)``."""
        enc = self.tokenize(examples)
        n_windows = len(enc["input_ids"])
        # Hay que leer los sequence_ids antes de hacer pop de los encodings.
        seq_ids = [enc.sequence_ids(i) for i in range(n_windows)]
        sample_map = enc.pop("overflow_to_sample_mapping")  # ventana -> índice del ejemplo original
        offsets = enc.pop("offset_mapping")
        return self.build(enc, sample_map, offsets, seq_ids, examples)

    def build(self, enc, sample_map, offsets, seq_ids, examples):
        """Lo que cambia entre entrenamiento y validación. Las subclases lo implementan."""
        raise NotImplementedError


class TrainFeatures(BaseQAFeatures):
    """Añade las etiquetas de inicio y fin en tokens a cada ventana."""

    def build(self, enc, sample_map, offsets, seq_ids, examples):
        starts, ends = [], []
        for w, ex_idx in enumerate(sample_map):
            answer = examples["answers"][ex_idx]
            if len(answer["text"]) == 0:
                # Pregunta sin respuesta: todas sus ventanas apuntan al [CLS].
                starts.append(CLS_INDEX)
                ends.append(CLS_INDEX)
                continue
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            s, e = char_span_to_token_span(offsets[w], seq_ids[w], start_char, end_char)
            starts.append(s)
            ends.append(e)
        enc["start_positions"] = starts
        enc["end_positions"] = ends
        return enc


class ValidationFeatures(BaseQAFeatures):
    """Guarda lo necesario para reconstruir la respuesta en la post-procesamiento."""

    def build(self, enc, sample_map, offsets, seq_ids, examples):
        enc["example_id"] = [examples["id"][ex_idx] for ex_idx in sample_map]
        # Solo los tokens del contexto (sequence_id == 1) conservan su offset.
        # Los de la pregunta y los especiales quedan en None: así la post-procesamiento
        # no puede cortar texto de la pregunta por accidente.
        enc["offset_mapping"] = [
            [off if sid == 1 else None for off, sid in zip(window_offsets, window_seq)]
            for window_offsets, window_seq in zip(offsets, seq_ids)
        ]
        return enc


def char_span_to_token_span(window_offsets, window_seq_ids, start_char, end_char):
    """Convierte un span de caracteres en índices de token dentro de una ventana.

    Devuelve ``(CLS_INDEX, CLS_INDEX)`` si el span no está completo en la ventana.
    Un span está completo si empieza y termina dentro del contexto de esta ventana.
    """
    ctx = [i for i, sid in enumerate(window_seq_ids) if sid == 1]
    if not ctx:
        return CLS_INDEX, CLS_INDEX
    ctx_first, ctx_last = ctx[0], ctx[-1]

    # Caracteres que cubre el contexto de esta ventana.
    window_start_char = window_offsets[ctx_first][0]
    window_end_char = window_offsets[ctx_last][1]
    if not (window_start_char <= start_char and end_char <= window_end_char):
        return CLS_INDEX, CLS_INDEX

    # Primer token que termina después del inicio: contiene el primer carácter de la respuesta.
    start_tok = ctx_first
    while window_offsets[start_tok][1] <= start_char:
        start_tok += 1

    # Último token que empieza antes del fin: contiene el último carácter de la respuesta.
    end_tok = ctx_last
    while window_offsets[end_tok][0] >= end_char:
        end_tok -= 1

    # Opción B (decidida con medición, ver docs/fase2_preprocesamiento.md):
    # si la respuesta empieza o termina a mitad de una subpalabra, el token completo
    # incluiría caracteres que no son la respuesta. Como la predicción se corta por
    # offsets, el texto recuperado no coincidiría con la referencia. Esa ventana
    # se trata como "sin respuesta" en lugar de enseñar una etiqueta incorrecta.
    if window_offsets[start_tok][0] != start_char or window_offsets[end_tok][1] != end_char:
        return CLS_INDEX, CLS_INDEX

    return start_tok, end_tok
