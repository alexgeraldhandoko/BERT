"""Tokenizer features for span training and cross-window prediction."""

from __future__ import annotations


def training_features(examples, tokenizer, max_length: int, doc_stride: int):
    tokenized = tokenizer(
        [question.strip() for question in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    starts, ends = [], []

    for feature_index, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[feature_index]
        answer = examples["answers"][sample_index][0]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = tokenized.sequence_ids(feature_index)
        context_positions = [i for i, segment in enumerate(sequence_ids) if segment == 1]
        cls_index = tokenized["input_ids"][feature_index].index(tokenizer.cls_token_id)
        first, last = context_positions[0], context_positions[-1]

        # An overflow window without the complete gold span is a no-answer window.
        if offsets[first][0] > start_char or offsets[last][1] < end_char:
            starts.append(cls_index)
            ends.append(cls_index)
            continue

        token_start = first
        while token_start <= last and offsets[token_start][0] <= start_char:
            token_start += 1
        token_end = last
        while token_end >= first and offsets[token_end][1] >= end_char:
            token_end -= 1
        starts.append(token_start - 1)
        ends.append(token_end + 1)

    tokenized["start_positions"] = starts
    tokenized["end_positions"] = ends
    return tokenized


def prediction_features(examples, tokenizer, max_length: int, doc_stride: int):
    tokenized = tokenizer(
        [question.strip() for question in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    for feature_index, sample_index in enumerate(sample_mapping):
        tokenized["example_id"].append(examples["id"][sample_index])
        sequence_ids = tokenized.sequence_ids(feature_index)
        tokenized["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_index] == 1 else None
            for token_index, offset in enumerate(tokenized["offset_mapping"][feature_index])
        ]
    return tokenized

