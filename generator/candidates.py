"""
Candidate generator.

Responsibilities:
1. Select a persona using persona weights.
2. Select mobility / office / both using that persona's `cats`.
3. Generate abilities and capability values.
4. Apply candidate-specific missingness.
5. Select one alternative for assistive-technology and communication fields.
6. Generate past occupation, skills, and CV text.
7. Run consistency checks.
8. Apply optional scripted edge cases.
9. Validate and write candidate JSON and private ground truth.

Run:
    python generator/candidates.py
    python generator/candidates.py . 100

Expected files:
    registry/registry.yaml
    registry/config.yaml
    registry/personas.yaml
    registry/job_templates.yaml        (optional but recommended)
    registry/edge_cases.yaml           (optional)
    schemas/candidate_v0_2.json
    generator/ability_model.py
"""

import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import yaml
from jsonschema import Draft7Validator

from ability_model import (
    rng_for,
    load_items,
    draw_abilities,
    generate_values,
    applies,
)


SCHEMA_VERSION = "0.2"


# ---------------------------------------------------------------------
# Default past occupations
#
# These are used only if a persona does not yet contain `past_occupations`.
# The names are matched against occupation titles in job_templates.yaml.
#
# It is better to add/adjust these lists in personas.yaml once the
# exact occupation-template titles have been reviewed.
# ---------------------------------------------------------------------

DEFAULT_PAST_OCCUPATIONS = {
    "wheelchair_user_lower_body": [
        "administrative assistant",
        "customer service representative",
        "data entry clerk",
        "accounting clerk",
    ],
    "wheelchair_user_arms_affected": [
        "telephone operator",
        "customer service representative",
        "receptionist",
        "administrative assistant",
    ],
    "lower_limb_amputation": [
        "sales assistant",
        "customer service representative",
        "cashier",
        "office clerk",
    ],
    "upper_limb_amputation": [
        "telephone operator",
        "customer service representative",
        "data entry clerk",
        "administrative assistant",
    ],
    "tremor": [
        "customer service representative",
        "receptionist",
        "administrative assistant",
        "sales representative",
    ],
    "cerebral_palsy": [
        "office clerk",
        "customer service representative",
        "library assistant",
        "administrative assistant",
    ],
    "one_sided_weakness": [
        "customer service representative",
        "office clerk",
        "sales assistant",
        "administrative assistant",
    ],
    "progressive_neuromuscular": [
        "administrative assistant",
        "customer service representative",
        "telephone operator",
        "office clerk",
    ],
    "skeletal_condition": [
        "office clerk",
        "customer service representative",
        "sales representative",
        "administrative assistant",
    ],
    "blind": [
        "telephone operator",
        "customer service representative",
        "call centre agent",
        "administrative assistant",
    ],
    "low_vision_central": [
        "customer service representative",
        "telephone operator",
        "administrative assistant",
        "office clerk",
    ],
    "low_vision_peripheral": [
        "customer service representative",
        "receptionist",
        "administrative assistant",
        "telephone operator",
    ],
    "multiple_conditions": [
        "customer service representative",
        "telephone operator",
        "administrative assistant",
        "office clerk",
    ],
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def load_yaml(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def weighted_choice(rng, values, weights):
    if not values:
        raise ValueError("Cannot choose from an empty list.")

    if len(values) != len(weights):
        raise ValueError("Values and weights must have equal lengths.")

    weights = [max(0.0, float(w)) for w in weights]

    if sum(weights) <= 0:
        raise ValueError("At least one weight must be positive.")

    return rng.choices(values, weights=weights, k=1)[0]


def choose_from_weight_map(rng, weights):
    if not weights:
        raise ValueError("Cannot choose from an empty weight map.")

    names = list(weights.keys())
    values = [float(weights[name]) for name in names]

    return weighted_choice(rng, names, values)


def choose_alternative(rng, alternatives, default=None):
    """
    Select one alternative from a persona field.

    Examples:
        [[keyboard, pointer, touch]]
            -> ["keyboard", "pointer", "touch"]

        [[voice, touch], [keyboard, voice]]
            -> one of the two lists

    If the field is already a flat list, it is returned as a list.
    """

    if alternatives is None:
        return list(default or [])

    if not isinstance(alternatives, list):
        return list(default or [])

    if not alternatives:
        return []

    # A list of lists means multiple alternatives.
    if all(isinstance(x, list) for x in alternatives):
        selected = rng.choice(alternatives)
        return list(selected)

    # A flat list represents one fixed alternative.
    return list(alternatives)


def choose_at_needs(rng, raw):
    """
    Expected persona format:
        at_needs: [[[wheelchair], [wheelchair, alternative_input]], 1.0]

    The first element contains alternative lists.
    The second element is the probability of having those needs.
    """

    if not raw:
        return []

    if (
        isinstance(raw, list)
        and len(raw) == 2
        and isinstance(raw[0], list)
        and isinstance(raw[1], (int, float))
    ):
        alternatives = raw[0]
        probability = float(raw[1])

        if rng.random() >= probability:
            return []

        if not alternatives:
            return []

        selected = rng.choice(alternatives)
        return list(selected)

    # Backward-compatible handling of a simple list.
    return choose_alternative(rng, raw)


def choose_disability_description(rng, persona):
    alternatives = persona.get("disability", [])

    if isinstance(alternatives, dict):
        alternatives = [alternatives]

    if not alternatives:
        return "Not specified"

    selected = rng.choice(alternatives)

    if isinstance(selected, dict):
        return str(selected.get("description", "Not specified"))

    return str(selected)


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


# ---------------------------------------------------------------------
# Persona and category selection
# ---------------------------------------------------------------------

def pick_persona(seed, index, personas):
    """Select a persona using its global weight."""

    if not isinstance(personas, dict) or not personas:
        raise ValueError("Personas configuration must be a non-empty dictionary.")

    names = list(personas.keys())
    weights = []

    for name in names:
        weight = personas[name].get("weight", 1)

        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(
                f"Invalid weight for persona '{name}': {weight}"
            )

        weights.append(weight)

    if sum(weights) <= 0:
        raise ValueError("Persona weights must have a positive total.")

    rng = rng_for(seed, "cand", index, "persona")

    return rng.choices(names, weights=weights, k=1)[0]


def pick_category(seed, index, persona):
    """Select mobility, office, or both using the selected persona's cats."""

    if not isinstance(persona, dict):
        raise TypeError(
            f"Expected a persona dictionary, got {type(persona).__name__}: "
            f"{persona!r}"
        )

    cats = persona.get("cats")

    if not isinstance(cats, dict) or not cats:
        raise ValueError(
            "Persona is missing a valid `cats` field. "
            f"Available persona fields: {list(persona.keys())}"
        )

    required_categories = {"mobility", "office", "both"}

    missing = required_categories - set(cats.keys())

    if missing:
        raise ValueError(
            f"Persona is missing category probabilities: {sorted(missing)}"
        )

    weights = [cats[name] for name in ("mobility", "office", "both")]

    if any(not isinstance(w, (int, float)) or w < 0 for w in weights):
        raise ValueError(f"Invalid category probabilities: {cats}")

    if sum(weights) <= 0:
        raise ValueError(f"Category probabilities must have a positive total: {cats}")

    rng = rng_for(seed, "cand", index, "category")

    return rng.choices(
        ["mobility", "office", "both"],
        weights=weights,
        k=1
    )[0]


# ---------------------------------------------------------------------
# Occupation templates and ESCO skills
# ---------------------------------------------------------------------

def find_job_templates_file(root):
    candidates = [
        root / "registry" / "job_templates.yaml",
        root / "registry" / "job_templates.yml",
        root / "job_templates.yaml",
        root / "job_templates.yml",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_job_templates(root):
    path = find_job_templates_file(root)

    if path is None:
        return {}, []

    data = load_yaml(path)

    templates = data.get("templates", {})

    if isinstance(templates, list):
        templates = {
            str(i): item
            for i, item in enumerate(templates)
        }

    if not isinstance(templates, dict):
        templates = {}

    general_skills = data.get("general_skills", []) or []

    return templates, general_skills


def template_title(key, template):
    return str(
        template.get("title")
        or template.get("occupation")
        or template.get("name")
        or key
    )


def template_category(template):
    value = (
        template.get("job_type")
        or template.get("category")
        or template.get("type")
    )

    if value in ("mobility", "office", "both"):
        return value

    return None


def occupation_similarity(a, b):
    """
    Compare occupation labels using both word overlap and string similarity.
    """

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    a_words = set(re.findall(r"[a-z0-9]+", a))
    b_words = set(re.findall(r"[a-z0-9]+", b))

    if not a_words or not b_words:
        return 0.0

    overlap = len(a_words & b_words) / len(a_words | b_words)
    sequence = SequenceMatcher(None, a, b).ratio()

    return 0.6 * overlap + 0.4 * sequence


def get_persona_occupation_options(persona_name, persona):
    """
    Prefer an explicit list in personas.yaml.

    Example:
        past_occupations:
          - administrative assistant
          - customer service representative
    """

    configured = persona.get("past_occupations")

    if configured:
        return list(configured)

    return DEFAULT_PAST_OCCUPATIONS.get(
        persona_name,
        ["administrative assistant", "customer service representative"],
    )


def choose_past_occupation(
    seed,
    index,
    persona_name,
    persona,
    category,
    templates,
):
    """
    Choose a plausible previous occupation and match it to a job template.

    The occupation title is written into private ground truth and CV text,
    not as an extra top-level candidate JSON field.
    """

    rng = rng_for(seed, "cand", index, "past_occupation")

    occupation_options = get_persona_occupation_options(
        persona_name,
        persona,
    )

    requested_occupation = rng.choice(occupation_options)

    if not templates:
        return requested_occupation, None, {}

    scored = []

    for key, template in templates.items():
        title = template_title(key, template)
        similarity = occupation_similarity(requested_occupation, title)

        category_value = template_category(template)

        # Prefer templates matching the candidate's category.
        if category_value == category:
            similarity += 0.15
        elif category_value is not None and category_value != "both":
            similarity -= 0.10

        scored.append((similarity, key, title, template))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return requested_occupation, None, {}

    best_score, best_key, best_title, best_template = scored[0]

    # If the mapping is weak, choose a category-compatible template.
    # This avoids falsely claiming that a completely unrelated occupation
    # is the same as the requested past occupation.
    if best_score < 0.15:
        compatible = [
            (key, title, template)
            for _, key, title, template in scored
            if template_category(template) in (None, category, "both")
        ]

        if compatible:
            best_key, best_title, best_template = rng.choice(compatible)

    return best_title, best_key, best_template


def normalize_skill_name(skill):
    """
    Accept skill entries as strings, dictionaries, or structured lists.

    This also protects against the nested-list error previously seen
    in jobs.py.
    """

    if isinstance(skill, str):
        return skill.strip() or None

    if isinstance(skill, dict):
        return normalize_skill_name(
            skill.get("name")
            or skill.get("label")
            or skill.get("skill")
        )

    if isinstance(skill, (list, tuple)):
        if not skill:
            return None

        return normalize_skill_name(skill[0])

    return None


def get_template_skills(template):
    if not template:
        return [], []

    skill_data = template.get("skills", {}) or {}

    essential = [
        name
        for item in skill_data.get("essential", []) or []
        if (name := normalize_skill_name(item))
    ]

    optional = [
        name
        for item in skill_data.get("optional", []) or []
        if (name := normalize_skill_name(item))
    ]

    return essential, optional


def build_skill_alias_map(general_skills):
    """
    The general_skills structure previously supplied is:

        [recognized_name, [synonyms], duty]

    It is used only to vary wording, not to introduce unrelated skills.
    """

    aliases = {}

    for item in general_skills:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        canonical = normalize_skill_name(item[0])

        if not canonical:
            continue

        aliases[normalize_text(canonical)] = [canonical]

        synonyms = item[1] or []

        if isinstance(synonyms, str):
            synonyms = [synonyms]

        for synonym in synonyms:
            if isinstance(synonym, str) and synonym.strip():
                aliases[normalize_text(canonical)].append(synonym.strip())

    return aliases


def make_spelling_variant(rng, name):
    """
    Introduce a small number of realistic spelling errors.

    This is intentionally low probability so the majority of skill names
    remain recognizable.
    """

    if len(name) < 5:
        return name

    operation = rng.choice(["drop", "swap", "duplicate"])
    position = rng.randint(1, len(name) - 2)

    if operation == "drop":
        return name[:position] + name[position + 1:]

    if operation == "swap":
        chars = list(name)
        chars[position], chars[position + 1] = (
            chars[position + 1],
            chars[position],
        )
        return "".join(chars)

    return name[:position] + name[position] + name[position:]


def vary_skill_wording(rng, skill_name, alias_map):
    """
    Examples:
        Microsoft Excel -> MS Excel
        communication -> communicating clearly

    A small proportion of entries may contain a typo.
    """

    name = skill_name.strip()

    aliases = alias_map.get(normalize_text(name), [])

    if aliases and rng.random() < 0.25:
        name = rng.choice(aliases)

    common_variants = {
        "microsoft excel": ["Excel", "MS Excel"],
        "microsoft word": ["MS Word", "Word"],
        "customer service": ["customer support", "customer service"],
        "data entry": ["data entry", "entering data"],
        "communication": ["communication", "communicating clearly"],
    }

    variants = common_variants.get(normalize_text(name))

    if variants and rng.random() < 0.30:
        name = rng.choice(variants)

    if rng.random() < 0.025:
        name = make_spelling_variant(rng, name)

    return name


def generate_candidate_skills(
    seed,
    index,
    template,
    general_skills,
    cfg,
):
    """
    Generate candidate skills from the past occupation's ESCO skills.

    Output schema:
        [{"name": "...", "level": 1-5}, ...]
    """

    rng = rng_for(seed, "cand", index, "skills")

    essential, optional = get_template_skills(template)

    if not essential and not optional:
        return []

    settings = cfg.get("candidate_skills", {})

    essential_range = settings.get("essential_count", [3, 6])
    optional_range = settings.get("optional_count", [1, 3])

    essential_count = rng.randint(
        int(essential_range[0]),
        int(essential_range[1]),
    )

    optional_count = rng.randint(
        int(optional_range[0]),
        int(optional_range[1]),
    )

    selected_essential = rng.sample(
        essential,
        min(essential_count, len(essential)),
    ) if essential else []

    selected_optional = rng.sample(
        optional,
        min(optional_count, len(optional)),
    ) if optional else []

    selected = list(dict.fromkeys(
        selected_essential + selected_optional
    ))

    alias_map = build_skill_alias_map(general_skills)

    experience_years = rng.choices(
        [0, 1, 2, 3, 5, 8, 12],
        weights=[0.08, 0.12, 0.15, 0.20, 0.20, 0.15, 0.10],
        k=1,
    )[0]

    skills = []

    for skill in selected:
        # Experience influences the likely proficiency level, but does not
        # force every skill to have the same level.
        base_level = min(5, 1 + experience_years // 3)
        level = int(round(base_level + rng.choice([-1, 0, 0, 0, 1])))
        level = max(1, min(5, level))

        skills.append({
            "name": vary_skill_wording(rng, skill, alias_map),
            "level": level,
        })

    rng.shuffle(skills)

    return skills


# ---------------------------------------------------------------------
# CV generation
# ---------------------------------------------------------------------

def cv_settings(cfg):
    """
    Defaults are deliberately conservative.
    These can be overridden in config.yaml.
    """

    return {
        "experience_probability": 0.85,
        "projects_probability": 0.45,
        "awards_probability": 0.20,
        "sparse_probability": 0.12,
        "no_text_probability": 0.04,
        **(cfg.get("cv_generation", {}) or {}),
    }


def generate_cv_text(
    seed,
    index,
    occupation,
    template,
    skills,
    cfg,
):
    """
    Generate CV sections without inserting disability-related information.

    The text uses occupation and skill information. It avoids claiming
    physical capacities such as lifting a specific weight or standing
    for a specific number of hours.
    """

    rng = rng_for(seed, "cand", index, "cv")

    settings = cv_settings(cfg)

    result = {
        "experience": None,
        "projects": None,
        "awards": None,
    }

    if rng.random() < settings["no_text_probability"]:
        return result

    sparse = rng.random() < settings["sparse_probability"]

    skill_names = [item["name"] for item in skills]

    if (
        skill_names
        and rng.random() < settings["experience_probability"]
    ):
        years = rng.choice([1, 2, 3, 4, 5, 7, 9, 12])

        selected_skills = skill_names[:min(3, len(skill_names))]
        skill_text = ", ".join(selected_skills)

        result["experience"] = (
            f"Worked as a {occupation} for approximately {years} years. "
            f"Experience includes {skill_text}. "
            "Worked with colleagues and followed assigned procedures."
        )

    if (
        not sparse
        and skill_names
        and rng.random() < settings["projects_probability"]
    ):
        project_skill = rng.choice(skill_names)

        result["projects"] = (
            f"Contributed to a work-related project involving "
            f"{project_skill}. Supported task organisation, "
            "documentation, and completion of assigned activities."
        )

    if (
        not sparse
        and rng.random() < settings["awards_probability"]
    ):
        awards = [
            "Received recognition for reliable task completion.",
            "Received a certificate of participation in workplace training.",
            "Recognised for teamwork and professional development.",
            "Completed an additional vocational training programme.",
        ]

        result["awards"] = rng.choice(awards)

    return result


# ---------------------------------------------------------------------
# WHODAS conversion
# ---------------------------------------------------------------------

def whodas_response_to_score(response):
    """
    Convert the original WHODAS response into the stored score.

    Original response:
        0 = None
        1 = Mild
        2 = Moderate
        3 = Severe
        4 = Extreme / cannot do

    Stored score:
        5 = no difficulty
        1 = extreme difficulty / cannot do

    IMPORTANT:
    ability_model.sample_score() already generates a stored 1-5 score.
    Therefore, this conversion must NOT be applied to those generated
    scores a second time.
    """

    if not isinstance(response, int) or response not in range(5):
        raise ValueError("WHODAS response must be an integer from 0 to 4.")

    return 5 - response


# ---------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------

def choose_quality_profile(seed, index, cfg):
    profiles = cfg.get("quality_profiles", {})

    if not profiles:
        return {
            "name": "default",
            "item_dropout": 0.02,
            "section_skip": 0.01,
        }

    names = list(profiles.keys())
    weights = [
        float(profiles[name].get("weight", 1.0))
        for name in names
    ]

    rng = rng_for(seed, "cand", index, "quality_profile")
    name = weighted_choice(rng, names, weights)

    return {
        "name": name,
        **profiles[name],
    }


def section_of(code):
    """
    PC01 -> PC
    OF10 -> OF
    WH_D2_1 -> WH_D2
    """

    if code.startswith("WH_"):
        return "WH_" + code.split("_")[1]

    return code[:2]


def build_capabilities(
    seed,
    index,
    values,
    methods,
    items,
    cfg,
    quality,
):
    """
    Apply item dropout, section skipping, tested-item dropout,
    and WHODAS work-item skipping.
    """

    rng_all = rng_for(seed, "cand", index, "missing", "all")

    all_null_probability = float(
        cfg.get("all_null_probability", 0.0)
    )

    all_null = rng_all.random() < all_null_probability

    sections = {section_of(code) for code in values}

    skipped_sections = {
        section
        for section in sections
        if rng_for(
            seed,
            "cand",
            index,
            "missing",
            "section",
            section,
        ).random() < float(quality.get("section_skip", 0.0))
    }

    tested_multiplier = float(
        cfg.get("tested_dropout_multiplier", 1.0)
    )

    whodas_work_skip = float(
        cfg.get("whodas_work_items_skip", 0.0)
    )

    capabilities = []

    for code in sorted(values):
        item = items[code]
        is_score = item["type"] == "score"

        if is_score:
            base_dropout = float(
                quality.get("item_dropout", 0.0)
            )
        else:
            base_dropout = float(
                quality.get("item_dropout", 0.0)
            )

            if methods.get(code) == "tested":
                base_dropout *= tested_multiplier

        dropped = (
            all_null
            or section_of(code) in skipped_sections
            or rng_for(
                seed,
                "cand",
                index,
                "missing",
                "item",
                code,
            ).random() < base_dropout
        )

        # WH_D5_5 to WH_D5_8 are work/school-related items.
        if code in {"WH_D5_5", "WH_D5_6", "WH_D5_7", "WH_D5_8"}:
            if rng_for(
                seed,
                "cand",
                index,
                "whodas_work_skip",
                code,
            ).random() < whodas_work_skip:
                dropped = True

        if is_score:
            capabilities.append({
                "item_code": code,
                "score": None if dropped else values[code],
            })

        elif dropped:
            capabilities.append({
                "item_code": code,
                "value": None,
            })

        else:
            capabilities.append({
                "item_code": code,
                "value": values[code],
                "unit": item.get("unit"),
                "method": methods[code],
            })

    return capabilities, all_null


# ---------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------

WHODAS_NUMERIC_PAIRS = {
    "WH_D2_1": ["PC12", "PC17", "PC24", "OF03"],
    "WH_D2_2": ["PC21"],
    "WH_D2_3": ["PC11", "PC20", "OF05"],
    "WH_D2_4": ["PC11", "PC20", "OF05"],
    "WH_D2_5": ["PC11", "PC20"],
    "WH_D3_1": ["OF07", "OF17"],
    "WH_D3_2": ["OF07", "OF17"],
    "WH_D3_3": ["OF07", "OF17"],
    "WH_D1_1": ["OF31"],
    "WH_D1_2": ["OF22", "OF24"],
    "WH_D1_3": ["OF21", "OF24", "OF32"],
    "WH_D1_4": ["OF33"],
    "WH_D1_5": ["OF25", "OF28"],
    "WH_D1_6": ["OF27", "OF28"],
}


def capability_maps(capabilities):
    numeric = {}
    scores = {}

    for item in capabilities:
        code = item["item_code"]

        if "value" in item:
            numeric[code] = item["value"]

        elif "score" in item:
            scores[code] = item["score"]

    return numeric, scores


def check_whodas_consistency(capabilities):
    """
    Flag strong contradictions between WHODAS and numeric items.

    These checks are intentionally conservative. They do not change values.
    They produce warnings in private ground truth for review.
    """

    numeric, scores = capability_maps(capabilities)
    warnings = []

    for whodas_code, numeric_codes in WHODAS_NUMERIC_PAIRS.items():
        score = scores.get(whodas_code)

        if score is None:
            continue

        if score > 2:
            continue

        for numeric_code in numeric_codes:
            value = numeric.get(numeric_code)

            if value is None:
                continue

            contradiction = False

            if numeric_code in {"PC12", "PC17", "PC24", "OF03"}:
                contradiction = value >= 90

            elif numeric_code in {"PC11", "PC20", "OF05"}:
                contradiction = value >= 1000

            elif numeric_code in {"OF07", "OF17"}:
                contradiction = value >= 16

            elif numeric_code == "OF31":
                contradiction = value >= 90

            elif numeric_code in {"OF22", "OF24", "OF28"}:
                contradiction = value >= 16

            elif numeric_code in {"OF21", "OF32", "OF33"}:
                contradiction = value <= 5

            elif numeric_code in {"OF25"}:
                contradiction = value >= 8

            elif numeric_code == "OF27":
                contradiction = value >= 120

            if contradiction:
                warnings.append(
                    f"Possible WHODAS contradiction: {whodas_code} "
                    f"has score {score}, but {numeric_code} is {value}."
                )

    return warnings


def check_cv_consistency(cv_text, capabilities):
    """
    Check generated CV text for explicit physical-capacity claims.

    The CV generator avoids these claims by design. This check is a
    safety net for future additions to the CV templates.
    """

    numeric, _ = capability_maps(capabilities)

    combined = " ".join(
        str(value)
        for value in cv_text.values()
        if value
    ).lower()

    warnings = []

    # Explicit claims of heavy lifting should be reviewed if the
    # candidate's measured lifting capacity is low.
    lifting_values = [
        numeric.get(code)
        for code in ("PC01", "PC02", "PC03", "PC04", "PC19")
        if numeric.get(code) is not None
    ]

    if lifting_values and min(lifting_values) <= 5:
        if re.search(
            r"\b(heavy lifting|lift heavy loads|manual loading)\b",
            combined,
        ):
            warnings.append(
                "CV mentions heavy lifting despite low lifting capability."
            )

    return warnings


def run_consistency_checks(cv_text, capabilities):
    return (
        check_whodas_consistency(capabilities)
        + check_cv_consistency(cv_text, capabilities)
    )


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def load_edge_cases(root):
    path = root / "registry" / "edge_cases.yaml"

    if not path.exists():
        path = root / "edge_cases.yaml"

    if not path.exists():
        return []

    data = load_yaml(path)

    if isinstance(data, list):
        return data

    return data.get("cases", []) or []


def apply_edge_case(clean, truth, edge_case, items):
    """
    Supported edge-case kinds:

        all_null_capabilities
        fits_nothing
        fits_everything
        empty_skills
        very_long_cv
        conflicting_values

    Example edge_cases.yaml:

        cases:
          - candidate_index: 301
            kind: all_null_capabilities

          - candidate_index: 302
            kind: fits_nothing

          - candidate_index: 303
            kind: fits_everything

          - candidate_index: 304
            kind: empty_skills

          - candidate_index: 305
            kind: very_long_cv

          - candidate_index: 306
            kind: conflicting_values

    The candidate_index is the numeric part of cand_XXXX.
    """

    kind = edge_case.get("kind")

    if kind == "all_null_capabilities":
        for capability in clean["capabilities"]:
            if "score" in capability:
                capability["score"] = None
            else:
                capability["value"] = None

    elif kind in {"fits_nothing", "fits_everything"}:
        fits_everything = kind == "fits_everything"

        for capability in clean["capabilities"]:
            code = capability["item_code"]

            if code.startswith("WH_"):
                capability["score"] = 5 if fits_everything else 1
                continue

            if code not in items:
                continue

            item = items[code]
            low, high = item.get("range", [0, 1])

            if item.get("hib", True):
                value = high if fits_everything else low
            else:
                value = low if fits_everything else high

            if "value" in capability:
                capability["value"] = value
                capability["unit"] = item.get("unit")
                capability["method"] = capability.get(
                    "method",
                    "self_reported",
                )

    elif kind == "empty_skills":
        clean["skills"] = []

    elif kind == "very_long_cv":
        long_text = (
            "Professional experience includes administrative support, "
            "task coordination, record keeping, customer communication, "
            "document preparation, and workplace collaboration. "
        ) * 80

        clean["cv_text"]["experience"] = long_text
        clean["cv_text"]["projects"] = long_text
        clean["cv_text"]["awards"] = long_text

    elif kind == "conflicting_values":
        # Deliberately create a contradiction for testing the checker.
        score_item = next(
            (
                x for x in clean["capabilities"]
                if x["item_code"] == "WH_D2_1"
            ),
            None,
        )

        numeric_item = next(
            (
                x for x in clean["capabilities"]
                if x["item_code"] == "PC24"
            ),
            None,
        )

        if score_item is not None:
            score_item["score"] = 1

        if numeric_item is not None:
            numeric_item["value"] = 240
            numeric_item["unit"] = "minutes"
            numeric_item["method"] = "self_reported"

    else:
        raise ValueError(f"Unknown edge-case kind: {kind}")

    truth.setdefault("edge_cases", []).append(kind)


# ---------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------

def build_candidate(index, reg, cfg, personas, items, templates,general_skills, edge_cases):

    seed = cfg.get("seed", 42)
    cid = f"cand_{index:04d}"

    # 1. Select a persona.
    pname = pick_persona(seed, index, personas)

    # 2. Retrieve the selected persona's configuration.
    persona = personas[pname]

    # 3. Select a category using this persona's category probabilities.
    category = pick_category(seed, index, persona)

    # 4. Generate abilities and capability values.
    abilities = draw_abilities(seed, index, persona, cfg)

    values, methods = generate_values(
        seed,
        index,
        category,
        abilities,
        items,
        reg["constraints_le"],
        cfg
    )
    # 3. Candidate missingness.
    quality = choose_quality_profile(seed, index, cfg)

    capabilities, all_null = build_capabilities(
        seed,
        index,
        values,
        methods,
        items,
        cfg,
        quality,
    )

    # 4. Select past occupation and its ESCO skills.
    occupation, occupation_key, occupation_template = (
        choose_past_occupation(
            seed,
            index,
            pname,
            persona,
            category,
            templates,
        )
    )

    skills = generate_candidate_skills(
        seed,
        index,
        occupation_template,
        general_skills,
        cfg,
    )

    # 5. Generate CV text.
    cv_text = generate_cv_text(
        seed,
        index,
        occupation,
        occupation_template,
        skills,
        cfg,
    )

    # 6. Select persona alternatives.
    prefs_rng = rng_for(seed, "cand", index, "prefs")

    disability_description = choose_disability_description(
        rng_for(seed, "cand", index, "disability"),
        persona,
    )

    at_needs = choose_at_needs(
        rng_for(seed, "cand", index, "at_needs"),
        persona.get("at_needs"),
    )

    input_methods = choose_alternative(
        rng_for(seed, "cand", index, "input_methods"),
        persona.get("input_methods"),
    )

    usable_formats = choose_alternative(
        rng_for(seed, "cand", index, "usable_formats"),
        persona.get("usable_formats"),
    )

    communication_channels = choose_alternative(
        rng_for(seed, "cand", index, "channels"),
        persona.get("channels"),
    )

    cities = cfg.get("cities", ["Cairo"])

    city_weights = cfg.get("city_weights")

    if city_weights and len(city_weights) == len(cities):
        location = weighted_choice(
            prefs_rng,
            cities,
            city_weights,
        )
    else:
        location = prefs_rng.choice(cities)

    remote_probability = float(
        persona.get(
            "remote_prob",
            cfg.get("remote_prob", 0.25),
        )
    )

    remote = prefs_rng.random() < remote_probability

    # 7. Public candidate JSON.
    clean = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": cid,
        "job_category": category,

        # This is the selected description, not the private persona name.
        "disability": disability_description,

        "skills": skills,
        "cv_text": cv_text,
        "capabilities": capabilities,

        "input_methods": input_methods,
        "usable_formats": usable_formats,
        "communication_channels": communication_channels,
        "at_needs": at_needs,

        "preferences": {
            "remote": remote,
            "location": location,
        },
    }

    # 8. Private ground truth.
    truth = {
        "candidate_id": cid,
        "persona": pname,
        "category": category,

        "past_occupation": occupation,
        "past_occupation_template_key": occupation_key,

        "quality_profile": quality["name"],
        "all_null": all_null,

        "abilities": {
            key: round(value, 3)
            for key, value in abilities.items()
        },

        # Complete values before missingness.
        "true_values": values,

        "true_methods": methods,

        "selected_disability_description": disability_description,

        "consistency_warnings": [],
    }

    # 9. Apply any scripted edge case assigned to this candidate.
    for edge_case in edge_cases:
        if int(edge_case.get("candidate_index", -1)) == index:
            apply_edge_case(
                clean,
                truth,
                edge_case,
                items,
            )

    # 10. Consistency checks.
    truth["consistency_warnings"] = run_consistency_checks(
        clean["cv_text"],
        clean["capabilities"],
    )

    return clean, truth


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def make_validator(root):
    schema_path = root / "schemas" / "candidate_v0_2.json"

    schema = json.loads(
        schema_path.read_text(encoding="utf-8")
    )

    Draft7Validator.check_schema(schema)

    return Draft7Validator(schema)


def validate_or_stop(validator, doc, items):
    problems = [
        f"{'/'.join(map(str, error.path)) or '(root)'}: {error.message}"
        for error in sorted(
            validator.iter_errors(doc),
            key=lambda error: list(error.path),
        )
    ]

    codes = [
        capability["item_code"]
        for capability in doc["capabilities"]
    ]

    if len(codes) != len(set(codes)):
        problems.append("capabilities: duplicate item_code")

    for code in codes:
        if code not in items:
            problems.append(
                f"capabilities: unknown item_code {code}"
            )

        elif not applies(items[code], doc["job_category"]):
            problems.append(
                f"capabilities: {code} is not allowed for "
                f"category {doc['job_category']}"
            )

    for skill in doc["skills"]:
        if not isinstance(skill.get("name"), str) or not skill["name"].strip():
            problems.append("skills: empty or invalid skill name")

        level = skill.get("level")

        if not isinstance(level, int) or not 1 <= level <= 5:
            problems.append(
                f"skills: invalid proficiency level {level}"
            )

    if problems:
        raise SystemExit(
            f"{doc['candidate_id']} failed validation:\n  "
            + "\n  ".join(problems[:20])
        )


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_all(root, count=None):
    root = Path(root)

    reg = load_yaml(root / "registry" / "registry.yaml")
    cfg = load_yaml(root / "registry" / "config.yaml")
    personas = load_yaml(root / "registry" / "personas.yaml")
    print("Loaded persona names:", list(personas.keys()))

    for name, data in personas.items():
        print(name, "->", list(data.keys()))

    templates, general_skills = load_job_templates(root)

    items = load_items(reg)
    validator = make_validator(root)

    edge_cases = load_edge_cases(root)

    if count is None:
        count = int(cfg.get("n_candidates", 50))

    if count < 1:
        raise ValueError("Candidate count must be at least 1.")

    # If edge cases use indices beyond the requested count, include them.
    edge_indices = [
        int(case["candidate_index"])
        for case in edge_cases
        if "candidate_index" in case
    ]

    if edge_indices:
        count = max(count, max(edge_indices))

    out = root / "out"
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Remove only previous generated candidate JSON files.
    # This prevents stale candidates from remaining after a smaller run.
    for old_file in candidate_dir.glob("cand_*.json"):
        old_file.unlink()

    truths = {}
    by_category = {}
    by_persona = {}
    by_occupation = {}

    for index in range(1, count + 1):
        clean, truth = build_candidate(
            index=index,
            reg=reg,
            cfg=cfg,
            personas=personas,
            items=items,
            templates=templates,
            general_skills=general_skills,
            edge_cases=edge_cases,
        )

        validate_or_stop(
            validator,
            clean,
            items,
        )

        cid = clean["candidate_id"]

        write_json(
            candidate_dir / f"{cid}.json",
            clean,
        )

        truths[cid] = truth

        category = truth["category"]
        persona = truth["persona"]
        occupation = truth["past_occupation"]

        by_category[category] = by_category.get(category, 0) + 1
        by_persona[persona] = by_persona.get(persona, 0) + 1
        by_occupation[occupation] = (
            by_occupation.get(occupation, 0) + 1
        )

    # Replace the previous files completely. Do not merge with old runs.
    write_json(
        out / "ground_truth.json",
        {
            "candidates": truths,
        },
    )

    write_json(
        out / "manifest.json",
        {
            "seed": cfg.get("seed", 42),
            "schema_version": SCHEMA_VERSION,
            "count": count,
            "by_category": by_category,
            "by_persona": by_persona,
            "by_past_occupation": by_occupation,
            "edge_case_count": len(edge_cases),
            "template_count": len(templates),
        },
    )

    print(f"Wrote {count} candidates -> {candidate_dir}")
    print("Categories:", by_category)
    print("Personas:", by_persona)
    print("Past occupations:", by_occupation)

    if not templates:
        print(
            "WARNING: job_templates.yaml was not found. "
            "Candidate skills will be empty until occupation templates "
            "are available."
        )

    warning_count = sum(
        len(truth["consistency_warnings"])
        for truth in truths.values()
    )

    print(
        f"Consistency warnings recorded in ground_truth.json: "
        f"{warning_count}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    if len(sys.argv) > 2:
        candidate_count = int(sys.argv[2])
    else:
        config = load_yaml(root / "registry" / "config.yaml")
        candidate_count = int(config.get("n_candidates", 50))

    generate_all(root, candidate_count)