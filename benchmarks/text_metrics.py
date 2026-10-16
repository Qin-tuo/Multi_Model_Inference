"""Frozen synthetic prompt-agreement CER, not real-speech accuracy."""

import unicodedata


def normalize(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text).lower()
        if not unicodedata.category(c).startswith(("P", "Z")) and not c.isspace()
    )


def character_errors(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return {
        "reference_normalized": reference,
        "hypothesis_normalized": hypothesis,
        "edit_distance": previous[-1],
        "reference_characters": len(reference),
        "cer": previous[-1] / len(reference) if reference else None,
    }
