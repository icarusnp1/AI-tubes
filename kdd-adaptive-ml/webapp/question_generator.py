import random

def _unique_choices(rng, correct, candidates, k=4, lo=None, hi=None):
    s = {correct}
    for c in candidates:
        if lo is not None and c < lo:
            continue
        if hi is not None and c > hi:
            continue
        s.add(c)
        if len(s) >= k:
            break
    while len(s) < k:
        jitter = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        s.add(correct + jitter)
    out = list(s)
    rng.shuffle(out)
    return out[:k]

def _fingerprint(template_id: str, params: dict) -> str:
    parts = [template_id] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(parts)

# --------------------
# LEVEL 1 (easy)
# --------------------
def gen_L1_eval_linear(rng):
    x = rng.randint(1, 20)
    a = rng.randint(1, 5)
    b = rng.randint(0, 10)
    ans = a * x + b
    params = {"x": x, "a": a, "b": b, "ans": ans}
    choices = _unique_choices(rng, ans, [ans+1, ans-1, a+b+x, (x+b)*a], lo=0, hi=200)
    return {
        "level": 1,
        "template_id": "L1_eval_linear",
        "concept_tag": "evaluate_expression",
        "prompt": f"Jika x = {x}, berapa nilai {a}x + {b}?",
        "choices": choices,
        "answer": ans,
        "hint_steps": [
            f"Substitusi x = {x} ke {a}x + {b}.",
            f"Hitung {a}×{x} dulu, lalu tambah {b}."
        ],
        "explanation": f"{a}x + {b} = {a}×{x} + {b} = {a*x} + {b} = {ans}.",
        "fingerprint": _fingerprint("L1_eval_linear", params),
        "params": params
    }

def gen_L1_sub(rng):
    x = rng.randint(5, 30)
    a = rng.randint(1, 10)
    if x < a:
        x = a + rng.randint(1, 10)
    ans = x - a
    params = {"x": x, "a": a, "ans": ans}
    choices = _unique_choices(rng, ans, [ans+1, ans-1, x+a, a-x], lo=-50, hi=100)
    return {
        "level": 1,
        "template_id": "L1_sub",
        "concept_tag": "evaluate_expression",
        "prompt": f"Jika x = {x}, berapa nilai x - {a}?",
        "choices": choices,
        "answer": ans,
        "hint_steps": [f"Ganti x dengan {x}.", f"Hitung {x} - {a}."],
        "explanation": f"x - {a} = {x} - {a} = {ans}.",
        "fingerprint": _fingerprint("L1_sub", params),
        "params": params
    }

def gen_L1_mul(rng):
    x = rng.randint(1, 20)
    a = rng.randint(2, 6)
    ans = a * x
    params = {"x": x, "a": a, "ans": ans}
    choices = _unique_choices(rng, ans, [ans+2, ans-2, a+x, x*x], lo=0, hi=200)
    return {
        "level": 1,
        "template_id": "L1_mul",
        "concept_tag": "multiply_variable",
        "prompt": f"Jika x = {x}, berapa nilai {a}x?",
        "choices": choices,
        "answer": ans,
        "hint_steps": [f"{a}x berarti {a} × x.", f"Hitung {a} × {x}."],
        "explanation": f"{a}x = {a}×{x} = {ans}.",
        "fingerprint": _fingerprint("L1_mul", params),
        "params": params
    }

def gen_L1_div(rng):
    a = rng.randint(2, 10)
    q = rng.randint(1, 10)
    x = a * q
    ans = q
    params = {"x": x, "a": a, "ans": ans}
    choices = _unique_choices(rng, ans, [ans+1, ans-1, x-a, x+a], lo=0, hi=100)
    return {
        "level": 1,
        "template_id": "L1_div",
        "concept_tag": "divide_variable",
        "prompt": f"Jika x = {x}, berapa nilai x / {a}?",
        "choices": choices,
        "answer": ans,
        "hint_steps": [f"Substitusi x = {x}.", f"Hitung {x} / {a}."],
        "explanation": f"x/{a} = {x}/{a} = {ans}.",
        "fingerprint": _fingerprint("L1_div", params),
        "params": params
    }

# --------------------
# LEVEL 2 (medium) range 1–50
# --------------------
def gen_L2_add(rng):
    x = rng.randint(1, 50)
    a = rng.randint(1, 20)
    b = x + a
    params = {"x": x, "a": a, "b": b}
    choices = _unique_choices(rng, x, [x+1, x-1, b-a+1, b+a], lo=-50, hi=100)
    return {
        "level": 2,
        "template_id": "L2_add",
        "concept_tag": "solve_linear_one_step_add",
        "prompt": f"Selesaikan: x + {a} = {b}. Nilai x adalah ...",
        "choices": choices,
        "answer": x,
        "hint_steps": [f"Kurangi {a} dari kedua sisi.", f"x = {b} - {a}"],
        "explanation": f"x + {a} = {b} → x = {b} - {a} = {x}.",
        "fingerprint": _fingerprint("L2_add", params),
        "params": params
    }

def gen_L2_sub(rng):
    a = rng.randint(1, 20)
    b = rng.randint(1, 50)
    x = b + a
    params = {"a": a, "b": b}
    choices = _unique_choices(rng, x, [x+1, x-1, b-a, b+a], lo=-50, hi=100)
    return {
        "level": 2,
        "template_id": "L2_sub",
        "concept_tag": "solve_linear_one_step_sub",
        "prompt": f"Selesaikan: x - {a} = {b}. Nilai x adalah ...",
        "choices": choices,
        "answer": x,
        "hint_steps": [f"Tambahkan {a} ke kedua sisi.", f"x = {b} + {a}"],
        "explanation": f"x - {a} = {b} → x = {b} + {a} = {x}.",
        "fingerprint": _fingerprint("L2_sub", params),
        "params": params
    }

def gen_L2_mul(rng):
    x = rng.randint(1, 50)
    a = rng.randint(2, 10)
    b = a * x
    params = {"a": a, "b": b}
    choices = _unique_choices(rng, x, [x+1, x-1, b-a, b//a+2], lo=-50, hi=200)
    return {
        "level": 2,
        "template_id": "L2_mul",
        "concept_tag": "solve_linear_one_step_mul",
        "prompt": f"Selesaikan: {a}x = {b}. Nilai x adalah ...",
        "choices": choices,
        "answer": x,
        "hint_steps": [f"Bagi kedua sisi dengan {a}.", f"x = {b} / {a}"],
        "explanation": f"{a}x = {b} → x = {b}/{a} = {x}.",
        "fingerprint": _fingerprint("L2_mul", params),
        "params": params
    }

def gen_L2_div(rng):
    a = rng.randint(2, 10)
    b = rng.randint(1, 50)
    x = a * b
    params = {"a": a, "b": b}
    choices = _unique_choices(rng, x, [x+a, x-a, b, a*b+1], lo=-50, hi=500)
    return {
        "level": 2,
        "template_id": "L2_div",
        "concept_tag": "solve_linear_one_step_div",
        "prompt": f"Selesaikan: x / {a} = {b}. Nilai x adalah ...",
        "choices": choices,
        "answer": x,
        "hint_steps": [f"Kalikan kedua sisi dengan {a}.", f"x = {b} × {a}"],
        "explanation": f"x/{a} = {b} → x = {b}×{a} = {x}.",
        "fingerprint": _fingerprint("L2_div", params),
        "params": params
    }

# --------------------
# LEVEL 3 (hard)
# --------------------
def gen_L3_dist_plus(rng):
    a = rng.randint(2, 6)
    b = rng.randint(1, 20)
    ans = f"{a}x + {a*b}"
    params = {"a": a, "b": b}
    choices = list({ans, f"{a}x + {b}", f"{a}x + {a+b}", f"x + {a*b}"})
    rng.shuffle(choices)
    return {
        "level": 3,
        "template_id": "L3_dist_plus",
        "concept_tag": "distributive_property",
        "prompt": f"Sederhanakan: {a}(x + {b}) = ...",
        "choices": choices,
        "answer": ans,
        "hint_steps": [f"Kalikan {a} ke x dan ke {b}.", f"{a}(x+{b}) = {a}x + {a*b}"],
        "explanation": f"{a}(x+{b}) = {a}x + {a}×{b} = {a}x + {a*b}.",
        "fingerprint": _fingerprint("L3_dist_plus", params),
        "params": params
    }

def gen_L3_dist_solve(rng):
    x = rng.randint(1, 20)
    a = rng.randint(2, 5)
    b = rng.randint(1, 10)
    c = a * (x + b)
    params = {"a": a, "b": b, "c": c}
    choices = _unique_choices(rng, x, [x+1, x-1, (c-a*b)//a, c-a], lo=-50, hi=100)
    return {
        "level": 3,
        "template_id": "L3_dist_solve",
        "concept_tag": "distributive_then_solve",
        "prompt": f"Selesaikan: {a}(x + {b}) = {c}. Nilai x adalah ...",
        "choices": choices,
        "answer": x,
        "hint_steps": ["Distribusi dulu agar tidak ada kurung.", f"{a}x + {a*b} = {c}", f"Kurangi {a*b}, lalu bagi {a}."],
        "explanation": f"{a}x + {a*b} = {c} → {a}x = {c-a*b} → x = {(c-a*b)}/{a} = {x}.",
        "fingerprint": _fingerprint("L3_dist_solve", params),
        "params": params
    }

def gen_L3_combine(rng):
    a = rng.randint(1, 10)
    b = rng.randint(1, 10)
    ans = f"{a+b}x"
    params = {"a": a, "b": b}
    choices = list({ans, f"{a}x", f"{b}x", f"{a*b}x"})
    rng.shuffle(choices)
    return {
        "level": 3,
        "template_id": "L3_combine",
        "concept_tag": "combine_like_terms",
        "prompt": f"Sederhanakan: {a}x + {b}x = ...",
        "choices": choices,
        "answer": ans,
        "hint_steps": ["Gabungkan koefisien karena suku sejenis.", f"({a}+{b})x = {a+b}x"],
        "explanation": f"{a}x + {b}x = ({a}+{b})x = {a+b}x.",
        "fingerprint": _fingerprint("L3_combine", params),
        "params": params
    }

def gen_L3_both_sides(rng):
    x = rng.randint(1, 20)
    a = rng.randint(1, 5)
    c = rng.randint(1, 5)
    while c == a:
        c = rng.randint(1, 5)
    b = rng.randint(0, 20)
    d = (a - c) * x + b  # ensures x is solution

    tries = 0
    while (d < 0 or d > 50) and tries < 50:
        x = rng.randint(1, 20)
        a = rng.randint(1, 5)
        c = rng.randint(1, 5)
        while c == a:
            c = rng.randint(1, 5)
        b = rng.randint(0, 20)
        d = (a - c) * x + b
        tries += 1

    params = {"a": a, "b": b, "c": c, "d": d}
    choices = _unique_choices(rng, x, [x+1, x-1, abs(a-c), d-b], lo=-50, hi=100)
    return {
        "level": 3,
        "template_id": "L3_both_sides",
        "concept_tag": "solve_linear_both_sides",
        "prompt": f"Selesaikan: {a}x + {b} = {c}x + {d}. Nilai x adalah ...",
        "choices": choices,
        "answer": x,
        "hint_steps": [f"Kurangi {c}x dari kedua sisi.", f"({a}-{c})x = {d-b}", "Bagi dengan (a-c)."],
        "explanation": f"{a}x + {b} = {c}x + {d} → ({a}-{c})x = {d-b} → x = {(d-b)}/{(a-c)} = {x}.",
        "fingerprint": _fingerprint("L3_both_sides", params),
        "params": params
    }

LEVEL_TEMPLATES = {
    1: [gen_L1_eval_linear, gen_L1_sub, gen_L1_mul, gen_L1_div],
    2: [gen_L2_add, gen_L2_sub, gen_L2_mul, gen_L2_div],
    3: [gen_L3_dist_plus, gen_L3_dist_solve, gen_L3_combine, gen_L3_both_sides],
}

def generate_question(level: int, seed=None):
    rng = random.Random(seed)
    gen = rng.choice(LEVEL_TEMPLATES[int(level)])
    return gen(rng)
