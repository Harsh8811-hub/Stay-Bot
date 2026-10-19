"""
Who reads the party size — the model or the regex — and in what order.

Why this is a module and not four lines inside ``understand()``
--------------------------------------------------------------
It was four lines inside ``understand()``, and they were an either/or with the
wrong side first. The regex branch came first and an ``elif`` gave the model its
turn only when the regex had found nothing. On 2026-09-24 that produced:

    "Check in: 12/11/26 check out 15/11/26 person 5+1"

The regex matched the ``26`` of the checkout year — sitting exactly where a
headcount belongs — and wrote ``adults=26, children=0``. The model read the same
sentence as ``adults=5, children=1``, which is what the guest asked for, and was
never consulted. A guest who wanted six people was told the bot had 26, then
that no room could seat them.

The inversion is one word of structure, which is precisely why it needs a test:
nothing about reading the function tells you which side wins unless you trace
the ``if``/``elif``, and the losing side fails silently. Inline in a 2100-line
function no test could reach it. Here it is a pure function over six values, and
``tests/test_pax_resolution.py`` pins the ordering directly.

The rule, stated once
---------------------
**The model is the reader; the regex is the fallback.** The model handles "5+1",
Hinglish, and spellings no pattern anticipates. The regex stays because it is
instant and exact on the shapes it does know, and because a model that returns
nothing (an outage, or a field it did not see) must not leave the field
unfilled. Preferring the reader over the fallback — and never letting the
fallback overrule a value the reader actually produced — is the whole rule.

One consequence worth knowing before you edit this. In the regex branch
``children`` is forced to ``0``, because the regex has no way to see a child
mixed into a pax phrase and a consistent reading is better than a stale one. In
the model branch ``children`` is written **only when the model saw one** — it is
never zeroed. That asymmetry is deliberate: the ``+1`` above is the child the
old zeroing used to discard.
"""

from __future__ import annotations

from typing import Any


def resolve_pax_counts(
    *,
    model_adults: Any = None,
    model_children: Any = None,
    regex_pax: Any = None,
    has_child_kw: bool = False,
    is_option_number_selection: bool = False,
    expected_action: Any = None,
) -> tuple[dict[str, Any], bool]:
    """
    The count fields to write, and whether the regex answered.

    Returns ``(updates, used_regex)``. The caller merges ``updates`` into its own
    dict; ``used_regex`` is returned separately because one regex-only field
    (``num_rooms``, read beside the pax phrase) belongs to the caller's scope and
    must not be written when the model answered.

    Pure: no I/O, no model, no state. Every branch below is reachable from a
    test with plain arguments.
    """
    updates: dict[str, Any] = {}

    # The model first. `expected_action == "need_child_ages"` is preserved from
    # the pre-2026-09-24 code: while the bot is collecting ages, a headcount the
    # model volunteered does not overwrite the party it is collecting for.
    if (
        model_adults is not None
        and not is_option_number_selection
        and expected_action != "need_child_ages"
    ):
        updates["adults"] = model_adults
        updates["effective_adults"] = model_adults
        if model_children is not None and not is_option_number_selection:
            updates["children"] = model_children
            updates["effective_children"] = model_children
            if model_children == 0:
                updates["child_ages"] = []
                updates["child_age_ask_count"] = 0
        return updates, False

    # The regex, only when the model returned no adults at all.
    if regex_pax is not None and not has_child_kw and not is_option_number_selection:
        updates["adults"] = regex_pax
        updates["effective_adults"] = regex_pax
        # Forced, and only in this branch — see the module docstring.
        updates["children"] = 0
        updates["effective_children"] = 0
        updates["child_ages"] = []
        updates["child_age_ask_count"] = 0
        return updates, True

    return updates, False
