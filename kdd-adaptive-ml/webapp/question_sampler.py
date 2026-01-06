import random
from question_generator import generate_question

def _seed_int(seed):
    if seed is None:
        return None
    if isinstance(seed, int):
        return seed
    return abs(hash(str(seed))) % (2**31 - 1)

def generate_set(level: int, set_size: int, focus_concepts: list[str], seen_fingerprints: set, seed=None):
    rng = random.Random(_seed_int(seed))
    out = []
    max_tries = 3000

    # 70% dari focus, sisanya non-focus (agar tidak monoton)
    focus_n = int(round(set_size * 0.7)) if focus_concepts else 0

    tries = 0
    while len(out) < set_size and tries < max_tries:
        tries += 1
        q = generate_question(level, seed=rng.randint(1, 10**9))
        fp = q.get("fingerprint")

        if fp and fp in seen_fingerprints:
            continue

        if focus_concepts:
            if len(out) < focus_n:
                if q.get("concept_tag") not in focus_concepts:
                    continue
            else:
                if q.get("concept_tag") in focus_concepts:
                    continue

        out.append(q)
        if fp:
            seen_fingerprints.add(fp)

    # fallback kalau generator space sempit
    if len(out) < set_size:
        while len(out) < set_size:
            out.append(generate_question(level, seed=rng.randint(1, 10**9)))

    rng.shuffle(out)
    return out, seen_fingerprints
