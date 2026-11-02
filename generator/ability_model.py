"""Layer: ability model.

Turns a persona into a full set of numeric answers for one candidate.
  1. draw a hidden ability (0-1) for each domain          -> draw_abilities
  2. sample every registry item around its domain ability  -> generate_values
  3. enforce within-person constraints (PC02 <= PC01 ...)  -> inside generate_values

It does NOT choose the persona or category, add missingness, or write JSON.
Those are other layers. Output here is a COMPLETE set of values (nothing missing).

Reproducibility: every random draw uses its own seeded stream built from
(seed, candidate index, purpose, item code). Adding or editing one item never
changes the values of any other item, and the same candidate id always gives
the same person.
"""
import random

DOMAINS = ["lifting", "walking", "standing", "posture", "sitting",
           "dexterity", "vision", "communication", "cognitive", "endurance"]


# ---------------------------------------------------------------- helpers
def rng_for(*parts):
    """Independent, repeatable random stream for one purpose."""
    return random.Random("|".join(str(p) for p in parts))


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(lo, hi, x, scale="lin"):
    """Position x (0-1) inside [lo, hi]. 'log' spreads values multiplicatively."""
    return lo * (hi / lo) ** x if scale == "log" else lo + (hi - lo) * x


def snap(v, step, lo, hi):
    """Round to the item's step, keep inside range, return int when whole."""
    v = clip(round(v / step) * step, lo, hi)
    return int(v) if float(v).is_integer() else round(v, 2)


# ---------------------------------------------------------------- items
def load_items(registry):
    """Normalise registry entries: apply defaults, add WHODAS items as 'score' items."""
    items = {}
    for code, spec in registry["candidate_items"].items():
        s = dict(spec)
        s.update(code=code, type="value")
        s.setdefault("scale", "lin")
        s.setdefault("step", 1)
        s.setdefault("hib", True)
        s.setdefault("method", "self_reported")
        items[code] = s
    for code, (domain, question) in registry["whodas"].items():
        items[code] = {"code": code, "type": "score", "domain": domain,
                       "question": question, "cat": "all"}
    return items


def applies(item, category):
    """A 'both' candidate answers everything. Others answer their own category + 'all'."""
    return item["cat"] == "all" or category == "both" or item["cat"] == category


# ---------------------------------------------------------------- step 1: abilities
def draw_abilities(seed, index, persona, cfg):
    """Hidden ability per domain, each between 0 and 1.

    - Domains the persona lists under `abilities` get a random value inside that range.
    - Every other domain uses the normal population draw: a mix of one person-wide
      'general factor' (so abilities partly move together) and a draw of its own.
    - 'general' (used by WHODAS participation items) is the average of all domains.
    """
    a, b = cfg["ability_beta"]
    w = cfg["general_factor_weight"]
    g = rng_for(seed, "cand", index, "general").betavariate(a, b)
    fixed = persona.get("abilities", {}) or {}
    ab = {}
    for d in DOMAINS:
        if d in fixed:
            lo, hi = fixed[d]
            ab[d] = rng_for(seed, "cand", index, "ability", d).uniform(lo, hi)
        else:
            own = rng_for(seed, "cand", index, "ability", d).betavariate(a, b)
            ab[d] = clip(w * g + (1 - w) * own, 0.02, 0.98)
    ab["general"] = sum(ab[d] for d in DOMAINS) / len(DOMAINS)
    return ab


# ---------------------------------------------------------------- step 2: sample items
def sample_value(item, latent, rng, noise):
    """One numeric answer. Higher ability -> better answer, 'better' meaning a higher
    number normally, or a lower number when the item has hib: false (e.g. seconds needed)."""
    x = clip(latent + rng.gauss(0, noise), 0.0, 1.0)
    if not item["hib"]:
        x = 1 - x
    lo, hi = item["range"]
    return snap(lerp(lo, hi, x, item["scale"]), item["step"], lo, hi)


def sample_score(latent, rng, noise):
    """WHODAS-style score 1-5 (5 = no difficulty, 1 = extreme / cannot do)."""
    x = clip(latent + rng.gauss(0, noise), 0.0, 1.0)
    return int(clip(round(1 + 4 * x), 1, 5))


def generate_values(seed, index, category, abilities, items, constraints_le, cfg):
    """Complete answers for one candidate.

    Returns (values, methods):
      values  : {item_code: number or 1-5 score}
      methods : {item_code: 'self_reported' | 'tested'} for numeric items
    """
    values, methods = {}, {}
    for code, item in items.items():
        if not applies(item, category):
            continue
        r = rng_for(seed, "cand", index, "item", code)
        latent = abilities[item["domain"]]
        if item["type"] == "score":
            values[code] = sample_score(latent, r, cfg["score_noise_sd"])
        else:
            values[code] = sample_value(item, latent, r, cfg["item_noise_sd"])
            m = item["method"]
            methods[code] = r.choice(["self_reported", "tested"]) if m == "either" else m

    # within-person constraints: values[small] must not exceed values[big]
    for small, big in constraints_le:
        if small in values and big in values and values[small] > values[big]:
            it = items[small]
            r = rng_for(seed, "cand", index, "fix", small)
            values[small] = snap(values[big] * r.uniform(0.6, 0.95),
                                 it["step"], it["range"][0], it["range"][1])
    return values, methods


# ---------------------------------------------------------------- quick look
if __name__ == "__main__":
    # python3 generator/ability_model.py [project_root] [persona_name] [category]
    import sys
    from pathlib import Path
    import yaml
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    load = lambda p: yaml.safe_load((root / p).read_text(encoding="utf-8"))
    reg, cfg, personas = load("registry/registry.yaml"), load("registry/config.yaml"), load("registry/personas.yaml")
    name = sys.argv[2] if len(sys.argv) > 2 else next(iter(personas))
    cat = sys.argv[3] if len(sys.argv) > 3 else "both"
    items = load_items(reg)
    for i in (1, 2, 3):
        ab = draw_abilities(cfg["seed"], i, personas[name], cfg)
        vals, _ = generate_values(cfg["seed"], i, cat, ab, items, reg["constraints_le"], cfg)
        print(f"--- candidate {i}, persona {name}, category {cat}")
        print("abilities:", {k: round(v, 2) for k, v in ab.items()})
        print({k: vals[k] for k in ("PC01", "PC02", "PC06", "PC11", "PC24", "OF01", "OF10", "OF18", "WH_D2_1") if k in vals})