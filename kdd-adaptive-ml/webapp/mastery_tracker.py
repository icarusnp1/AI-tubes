def init_mastery(concepts: list[str]) -> dict:
    return {c: 0.5 for c in concepts}  # start neutral

def update_mastery(mastery: dict, concept_tag: str, is_correct: bool, used_hint: bool = False) -> dict:
    if not concept_tag:
        return mastery
    cur = float(mastery.get(concept_tag, 0.5))

    # correct gives +gain, wrong gives -penalty
    if is_correct:
        delta = 0.08 if not used_hint else 0.05
    else:
        delta = -0.12

    mastery[concept_tag] = max(0.0, min(1.0, cur + delta))
    return mastery

def weakest_concepts(mastery: dict, k: int = 2) -> list[str]:
    items = sorted(mastery.items(), key=lambda kv: kv[1])
    return [c for c, _ in items[:k]]
