"""Translate LlamaIndex ``MetadataFilters`` into TopK filter expressions."""

from __future__ import annotations

from typing import Any

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from topk_sdk.query import LogicalExpr, field, not_
from topk_sdk.query import all as topk_all
from topk_sdk.query import any as topk_any

METADATA_FIELD = "metadata"
REF_DOC_ID_FIELD = "ref_doc_id"

# LlamaIndex code filters on the source document under any of these names.
_REF_DOC_KEYS = {"ref_doc_id", "doc_id", "document_id"}


def translate_filters(
    filters: MetadataFilters | LogicalExpr | None,
) -> LogicalExpr | None:
    """
    Translate filters into a TopK expression.

    Accepts LlamaIndex ``MetadataFilters`` (nested ``MetadataFilters`` and the
    ``AND``/``OR``/``NOT`` conditions included) or a native TopK expression, which
    is passed through unchanged.

    Metadata keys address the node's metadata; nested values use dots
    (``"author.name"``). ``ref_doc_id``, ``doc_id`` and ``document_id`` match the
    node's source document.
    """
    if filters is None:
        return None
    if isinstance(filters, LogicalExpr):
        return filters
    if not isinstance(filters, MetadataFilters):
        msg = f"filters must be MetadataFilters or a TopK expression, got {type(filters).__name__}"
        raise TypeError(msg)
    if not filters.filters:
        return None

    clauses = [translate_filters(f) if isinstance(f, MetadataFilters) else _translate_one(f) for f in filters.filters]
    clauses = [c for c in clauses if c is not None]
    if not clauses:
        return None

    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.OR:
        return clauses[0] if len(clauses) == 1 else topk_any(clauses)
    combined = clauses[0] if len(clauses) == 1 else topk_all(clauses)
    if condition == FilterCondition.NOT:
        return not_(combined)
    return combined


def _field(key: str) -> Any:
    if key in _REF_DOC_KEYS:
        return field(REF_DOC_ID_FIELD)
    return field(f"{METADATA_FIELD}.{key}")


def _translate_one(f: MetadataFilter) -> LogicalExpr:
    expr = _field(f.key)
    op, value = f.operator, f.value

    if op == FilterOperator.EQ:
        return expr.eq(value)
    if op == FilterOperator.NE:
        return expr.ne(value)
    if op == FilterOperator.GT:
        return expr.gt(value)
    if op == FilterOperator.GTE:
        return expr.gte(value)
    if op == FilterOperator.LT:
        return expr.lt(value)
    if op == FilterOperator.LTE:
        return expr.lte(value)
    if op == FilterOperator.IN:
        return expr.in_(_as_list(f))
    if op == FilterOperator.NIN:
        return not_(expr.in_(_as_list(f)))
    if op == FilterOperator.CONTAINS:
        return expr.contains(value)
    if op == FilterOperator.ANY:
        return topk_any([expr.contains(v) for v in _as_list(f)])
    if op == FilterOperator.ALL:
        return topk_all([expr.contains(v) for v in _as_list(f)])
    if op == FilterOperator.TEXT_MATCH:
        # Substring match on a string metadata value.
        return expr.contains(value)
    if op == FilterOperator.IS_EMPTY:
        return expr.is_null()

    msg = (
        f"Filter operator {op.value!r} on {f.key!r} isn't supported by TopK. "
        "Pass a native TopK expression (topk_sdk.query) as filters instead."
    )
    raise ValueError(msg)


def _as_list(f: MetadataFilter) -> list[Any]:
    if not isinstance(f.value, list) or not f.value:
        msg = f"Filter operator {f.operator.value!r} on {f.key!r} expects a non-empty list"
        raise ValueError(msg)
    return list(f.value)
