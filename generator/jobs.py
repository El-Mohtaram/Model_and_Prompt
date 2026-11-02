
"""
Synthetic job generator for the M2 -> M3 job schema.

Reads:
    registry/config.yaml
    registry/registry.yaml
    registry/job_templates.yaml
    schemas/job_m2.json

Writes:
    out/jobs/<job_id>_v<version>.json
    out/ground_truth.json          (jobs section)
    out/manifest.json              (jobs section)

Run from the project root:
    python generator/jobs.py

Dependencies:
    pip install pyyaml jsonschema
"""

import copy
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml
from jsonschema import Draft7Validator

from ability_model import rng_for


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

JOB_STATUSES = ("published", "paused", "closed", "archived")

FREQUENCY_BANDS = (
    "never",
    "occasional",
    "frequent",
    "constant",
)

# The level ranges are synthetic intensity bands.
# Level 0 = very light; level 3 = demanding.
LEVEL_INTERVALS = {
    0: (0.05, 0.25),
    1: (0.20, 0.50),
    2: (0.40, 0.80),
    3: (0.65, 1.00),
}

DEFAULT_WORKPLACE_UNKNOWN_RATE = 0.15
DEFAULT_INCOMPLETE_JOB_SHARE = 0.12
DEFAULT_NO_ESSENTIAL_SHARE = 0.03

# These are synthetic salary assumptions, not sourced salary statistics.
# Override them in config.yaml if you have better estimates.
DEFAULT_SALARY_RANGES = {
    "mobility": (6000, 18000),
    "office": (8000, 30000),
    "both": (7000, 25000),
}

DEFAULT_WORK_MODES = {
    "mobility": {"on-site": 0.90, "hybrid": 0.08, "remote": 0.02},
    "office": {"on-site": 0.55, "hybrid": 0.30, "remote": 0.15},
    "both": {"on-site": 0.75, "hybrid": 0.20, "remote": 0.05},
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def load_yaml(path):
    """Load a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path, obj):
    """Write JSON, creating parent directories when necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def merge_json_section(path, section, values):
    """Update one section of a JSON file without deleting other sections."""
    path = Path(path)

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}

    data[section] = values
    write_json(path, data)


def clip(value, low, high):
    return max(low, min(high, value))


def snap(value, step, low, high):
    """Round a numeric value to its configured step."""
    if not step:
        return round(clip(value, low, high), 2)

    value = round(value / step) * step
    value = clip(value, low, high)

    if float(value).is_integer():
        return int(value)

    return round(value, 2)


def weighted_choice(rng, weights):
    """
    Select a key using a dictionary of weights.

    Example:
        {"published": 0.75, "paused": 0.08, ...}
    """
    if not weights:
        raise ValueError("Cannot select from an empty weight dictionary.")

    keys = list(weights.keys())
    values = [max(0, float(weights[k])) for k in keys]

    if sum(values) <= 0:
        return rng.choice(keys)

    return rng.choices(keys, weights=values, k=1)[0]


def sample_number(rng, low, high, step=1, scale="lin"):
    """Sample a number from a linear or logarithmic range."""
    if high < low:
        low, high = high, low

    if scale == "log" and low > 0:
        value = low * (high / low) ** rng.random()
    else:
        value = rng.uniform(low, high)

    return snap(value, step, low, high)


def choose_city(rng, cfg):
    """Choose a city, using configured weights when their lengths match."""
    cities = cfg.get("cities") or ["Cairo"]

    weights = cfg.get("city_weights")

    if weights and len(weights) == len(cities) and sum(weights) > 0:
        return rng.choices(cities, weights=weights, k=1)[0]

    # The supplied config currently has ten cities but six weights.
    return rng.choice(cities)


def make_date(rng):
    """
    Generate a reproducible measurement date.

    A fixed date range is used so regenerating with the same seed
    does not change the output just because the current date changed.
    """
    start = date(2024, 1, 1)
    days = rng.randint(0, 730)
    return (start + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------

def get_templates(template_data):
    """Return the occupation templates dictionary."""
    templates = template_data.get("templates", {})

    if not templates:
        raise ValueError(
            "No occupation templates were found in job_templates.yaml."
        )

    return templates


def select_template(seed, index, templates, cfg):
    """
    Select an occupation template.

    First, choose the desired job category using job_type_weights.
    Then choose a template belonging to that category.
    """
    category_rng = rng_for(seed, "job", index, "category")

    category = weighted_choice(
        category_rng,
        cfg.get(
            "job_type_weights",
            {"mobility": 0.35, "office": 0.40, "both": 0.25},
        ),
    )

    allowed = {
        name: template
        for name, template in templates.items()
        if template.get("type") == category
    }

    if not allowed:
        raise ValueError(
            f"No job templates found for category '{category}'."
        )

    template_rng = rng_for(seed, "job", index, "template")
    template_name = template_rng.choice(sorted(allowed))

    return template_name, allowed[template_name], category


# ---------------------------------------------------------------------
# Demand sampling
# ---------------------------------------------------------------------

def get_level(template, code):
    """
    Get a demand intensity level from the occupation template.

    D06 is the maximum of the sitting and standing levels when
    those levels are available.
    """
    levels = template.get("levels", {}) or {}

    if code == "D06":
        relevant = [
            levels[c]
            for c in ("D04", "D05")
            if c in levels
        ]

        if relevant:
            return max(relevant)

    return int(levels.get(code, 1))


def sample_scaled_demand(rng, spec, level):
    """
    Sample a scaled demand according to the occupation's intensity.

    The registry provides the actual range, unit, scale and step.
    The occupation template provides the intensity level.
    """
    low, high = spec["range"]

    level = int(clip(level, 0, 3))
    fraction_low, fraction_high = LEVEL_INTERVALS[level]

    actual_low = low + (high - low) * fraction_low
    actual_high = low + (high - low) * fraction_high

    # D01's typical_max is a registry-provided cap for typical jobs.
    if spec.get("typical_max") is not None:
        actual_high = min(actual_high, spec["typical_max"])

    actual_high = max(actual_low, actual_high)

    return sample_number(
        rng,
        actual_low,
        actual_high,
        step=spec.get("step", 1),
        scale=spec.get("scale", "lin"),
    )


def sample_multi_value(rng, spec, template, code):
    """
    Sample a multi-select demand.

    Registry options provide default probabilities.
    A template may override individual probabilities.
    """
    options = copy.deepcopy(spec.get("options", {}) or {})

    overrides = (
        template.get("multi", {}).get(code, {})
        if template.get("multi")
        else {}
    )

    options.update(overrides)

    selected = [
        option
        for option, probability in options.items()
        if rng.random() < float(probability)
    ]

    # Avoid an empty known multi-select value.
    if not selected and options:
        selected = [weighted_choice(rng, options)]

    return sorted(selected)


def sample_environmental_value(rng, code, spec):
    """Sample environmental demand values from the registry."""
    kind = spec["kind"]

    if kind == "env_uniform":
        low, high = spec["range"]

        return sample_number(
            rng,
            low,
            high,
            step=spec.get("step", 1),
        )

    if kind == "env_noise":
        # Typical and peak noise levels, in dB(A).
        typical = round(rng.uniform(35, 85), 1)
        peak = round(typical + rng.uniform(3, 15), 1)

        return {
            "typical": typical,
            "peak": peak,
        }

    if kind == "env_temp":
        # Lowest and highest workplace temperatures, in Celsius.
        lowest = round(rng.uniform(10, 28), 1)
        highest = round(rng.uniform(max(lowest + 2, 25), 42), 1)

        return {
            "lowest": lowest,
            "highest": highest,
        }

    raise ValueError(
        f"Unsupported environmental demand kind '{kind}' for {code}."
    )


def sample_demand_value(rng, code, spec, template):
    """Sample a known value for one demand."""
    kind = spec["kind"]

    if kind == "scaled":
        return sample_scaled_demand(
            rng,
            spec,
            get_level(template, code),
        )

    if kind == "yesno":
        p_no = template.get("d08_p_no", spec.get("p_no", 0.35))
        return rng.random() >= float(p_no)

    if kind == "multi":
        return sample_multi_value(rng, spec, template, code)

    if kind in ("env_uniform", "env_noise", "env_temp"):
        return sample_environmental_value(rng, code, spec)

    raise ValueError(
        f"Unsupported demand kind '{kind}' for {code}."
    )


def sample_frequency(rng, band_setting, job_type):
    """
    Determine whether a frequency band should be attached.

    Registry values:
        always     -> attach a band to every applicable job type
        both_only  -> attach a band only to Both jobs
        none       -> no frequency band
    """
    if band_setting == "none":
        return None

    if band_setting == "both_only" and job_type != "both":
        return None

    # Synthetic distribution of how frequently a task occurs.
    return weighted_choice(
        rng,
        {
            "never": 0.05,
            "occasional": 0.35,
            "frequent": 0.40,
            "constant": 0.20,
        },
    )


def sample_duration(rng, frequency_band):
    """Generate a duration in minutes for a demand."""
    if frequency_band is None or frequency_band == "never":
        return None

    ranges = {
        "occasional": (1, 30),
        "frequent": (15, 120),
        "constant": (30, 240),
    }

    low, high = ranges[frequency_band]
    return rng.randint(low, high)


def build_demands(seed, index, template, job_type, registry, cfg):
    """Generate all applicable D01-D17 demands for one job."""
    demands = []

    registry_demands = registry.get("job_demands", {})
    skipped = set(template.get("skip", []) or [])

    sloppy_share = float(cfg.get("sloppy_job_share", 0.15))
    sloppy_multiplier = float(cfg.get("sloppy_unknown_multiplier", 3.5))

    # One employer-level quality draw: a sloppy employer has a higher
    # chance of answering "Not sure" across multiple questions.
    employer_rng = rng_for(seed, "job", index, "employer_quality")
    sloppy_employer = employer_rng.random() < sloppy_share

    no_essential_share = float(
        cfg.get("no_essential_job_share", DEFAULT_NO_ESSENTIAL_SHARE)
    )
    no_essential = (
        rng_for(seed, "job", index, "no_essential").random()
        < no_essential_share
    )

    for code, spec in registry_demands.items():
        if code in skipped:
            continue

        applies_to = spec.get("applies", ["mobility", "office", "both"])

        if job_type not in applies_to:
            continue

        value_rng = rng_for(seed, "job", index, "demand", code)
        unknown_rng = rng_for(seed, "job", index, "unknown", code)
        essential_rng = rng_for(seed, "job", index, "essential", code)
        frequency_rng = rng_for(seed, "job", index, "frequency", code)
        duration_rng = rng_for(seed, "job", index, "duration", code)

        p_unknown = float(spec.get("p_unknown", 0.0))

        if sloppy_employer:
            p_unknown *= sloppy_multiplier

        p_unknown = clip(p_unknown, 0.0, 0.95)

        is_unknown = unknown_rng.random() < p_unknown

        frequency_band = sample_frequency(
            frequency_rng,
            spec.get("band", "none"),
            job_type,
        )

        duration_min = sample_duration(
            duration_rng,
            frequency_band,
        )

        demand = {
            "dimension_code": code,
            "value_state": "unknown" if is_unknown else "known",
            "value": None,
            "unit": spec.get("unit"),
            "frequency_band": frequency_band,
            "duration_min": duration_min,
        }

        if not is_unknown:
            demand["value"] = sample_demand_value(
                value_rng,
                code,
                spec,
                template,
            )

        # The M2 schema supports this optional field.
        # Unknown demands are not marked essential because their
        # importance has not been confirmed by the employer.
        if cfg.get("emit_essential", True):
            p_essential = float(spec.get("p_essential", 0.0))

            demand["essential"] = (
                False
                if is_unknown or no_essential
                else essential_rng.random() < p_essential
            )

        demands.append(demand)

    return demands, {
        "sloppy_employer": sloppy_employer,
        "no_essential_demands": no_essential,
    }


# ---------------------------------------------------------------------
# Workplace facts
# ---------------------------------------------------------------------

def sample_workplace_value(rng, spec):
    """Generate a workplace fact using its registry definition."""
    kind = spec["kind"]

    if kind == "bool":
        # Most workplaces have the basic facility, but not all.
        return rng.random() < 0.75

    if kind == "num":
        low, high = spec["range"]

        return sample_number(
            rng,
            low,
            high,
            step=spec.get("step", 1),
        )

    raise ValueError(f"Unsupported workplace kind: {kind}")


def build_workplace(seed, index, job_type, registry, cfg):
    """Generate applicable E-code workplace facts."""
    workplace = []

    items = registry.get("workplace_items", {})

    verification_weights = registry.get(
        "verification_weights",
        {
            "self_reported": 0.60,
            "evidenced": 0.25,
            "third_party_audited": 0.10,
            "community_confirmed": 0.05,
        },
    )

    unknown_rate = float(
        cfg.get(
            "workplace_unknown_rate",
            DEFAULT_WORKPLACE_UNKNOWN_RATE,
        )
    )

    sloppy_share = float(cfg.get("sloppy_job_share", 0.15))
    sloppy_multiplier = float(cfg.get("sloppy_unknown_multiplier", 3.5))

    quality_rng = rng_for(seed, "job", index, "employer_quality")
    sloppy_employer = quality_rng.random() < sloppy_share

    if sloppy_employer:
        unknown_rate *= sloppy_multiplier

    unknown_rate = clip(unknown_rate, 0.0, 0.95)

    for code, spec in items.items():
        applies_to = spec.get("applies", ["mobility", "office", "both"])

        if job_type not in applies_to:
            continue

        unknown_rng = rng_for(seed, "job", index, "workplace_unknown", code)
        value_rng = rng_for(seed, "job", index, "workplace_value", code)
        verification_rng = rng_for(
            seed, "job", index, "workplace_verification", code
        )
        date_rng = rng_for(seed, "job", index, "workplace_date", code)

        is_unknown = unknown_rng.random() < unknown_rate

        item = {
            "item_code": code,
            "value_state": "unknown" if is_unknown else "known",
            "value": None,
            "unit": spec.get("unit"),
            "verification_level": None,
            "measured_at": None,
        }

        if not is_unknown:
            item["value"] = sample_workplace_value(value_rng, spec)
            item["verification_level"] = weighted_choice(
                verification_rng,
                verification_weights,
            )
            item["measured_at"] = make_date(date_rng)

        workplace.append(item)

    return workplace


# ---------------------------------------------------------------------
# Skills and job text
# ---------------------------------------------------------------------

def title_for_template(rng, template):
    titles = template.get("titles") or ["Employee"]
    return rng.choice(titles)



def normalize_skill_name(skill):
    """Convert a skill entry into one canonical string name."""
    if isinstance(skill, str):
        return skill.strip() or None

    if isinstance(skill, dict):
        return normalize_skill_name(skill.get("name"))

    if isinstance(skill, (list, tuple)):
        if not skill:
            return None

        # ESCO-style structure:
        # [recognized name, synonyms, duty sentence]
        return normalize_skill_name(skill[0])

    raise TypeError(
        f"Unsupported skill format: {type(skill).__name__}: {skill!r}"
    )


def build_skills(seed, index, template):
    """Convert template skills into job-schema skill objects."""
    skill_data = template.get("skills", {}) or {}

    essential = [
        name for item in (skill_data.get("essential") or [])
        if (name := normalize_skill_name(item))
    ]

    optional = [
        name for item in (skill_data.get("optional") or [])
        if (name := normalize_skill_name(item))
    ]

    essential_rng = rng_for(seed, "job", index, "essential_skills")
    optional_rng = rng_for(seed, "job", index, "optional_skills")

    essential_count = (
        essential_rng.randint(min(4, len(essential)), min(8, len(essential)))
        if essential else 0
    )

    optional_count = (
        optional_rng.randint(min(1, len(optional)), min(3, len(optional)))
        if optional else 0
    )

    selected_essential = (
        essential_rng.sample(essential, essential_count)
        if essential_count else []
    )

    selected_optional = (
        optional_rng.sample(optional, optional_count)
        if optional_count else []
    )

    return (
        [
            {"name": name, "importance": "essential"}
            for name in selected_essential
        ]
        +
        [
            {"name": name, "importance": "preferred"}
            for name in selected_optional
        ]
    )

def build_duty_sentence(rng, skill_names):
    """Create a short paraphrased duty sentence from selected skill names."""
    if not skill_names:
        return None

    skill = rng.choice(skill_names)

    patterns = [
        f"Carry out tasks involving {skill} as part of the normal work routine.",
        f"Use practical knowledge of {skill} to complete assigned work.",
        f"Take responsibility for activities related to {skill}.",
        f"Apply {skill} when handling day-to-day responsibilities.",
        f"Support the team by completing work that requires {skill}.",
    ]

    return rng.choice(patterns)


def build_job_text(
    seed,
    index,
    title,
    template,
    skills,
    company,
    salary,
    cfg,
):
    """Generate the summary and description for a job posting."""
    text_rng = rng_for(seed, "job", index, "text")

    template_data = cfg.get("_template_data", {})

    openings = template_data.get(
        "openings",
        ["We are looking for a"],
    )
    company_lines = template_data.get(
        "company_lines",
        ["We offer a supportive working environment."],
    )

    opening = text_rng.choice(openings)
    company_line = text_rng.choice(company_lines)


    essential_names = [
        normalize_skill_name(item.get("name"))
        for item in skills
        if item.get("importance") == "essential"
    ]

    essential_names = [
        name for name in essential_names if name
    ]

    duties = []
    duty_rng = rng_for(seed, "job", index, "duties")

    selected_duties = duty_rng.sample(
        essential_names,
        k=min(3, len(essential_names))
    )

    for skill in selected_duties:
        sentence = build_duty_sentence(duty_rng, [skill])
        if sentence:
            duties.append(sentence)


    # Use a few skills to create paraphrased responsibilities.
    for _ in range(min(3, len(essential_names))):
        sentence = build_duty_sentence(duty_rng, essential_names)
        if sentence and sentence not in duties:
            duties.append(sentence)

    salary_text = (
        f"Monthly salary: EGP {salary[0]:,}–{salary[1]:,}."
    )

    summary = (
        f"{title} position at {company}, based in Egypt. "
        f"The role involves {', '.join(essential_names[:3])}."
        if essential_names
        else f"{title} position at {company}, based in Egypt."
    )

    description_parts = [
        f"{opening} {title}.",
        company_line,
    ]

    if duties:
        description_parts.append("Main responsibilities: " + " ".join(duties))

    description_parts.append(salary_text)

    incomplete_share = float(
        cfg.get(
            "incomplete_job_share",
            DEFAULT_INCOMPLETE_JOB_SHARE,
        )
    )

    incomplete = (
        rng_for(seed, "job", index, "incomplete_posting").random()
        < incomplete_share
    )

    if incomplete:
        # Some postings intentionally contain less information.
        if text_rng.random() < 0.5:
            summary = None
        else:
            description_parts = description_parts[:2]

    description = " ".join(description_parts)

    return summary, description, incomplete


# ---------------------------------------------------------------------
# Basic job fields
# ---------------------------------------------------------------------

def choose_status(seed, index, cfg):
    rng = rng_for(seed, "job", index, "status")

    return weighted_choice(
        rng,
        cfg.get(
            "job_status_weights",
            {
                "published": 0.75,
                "paused": 0.08,
                "closed": 0.10,
                "archived": 0.07,
            },
        ),
    )


def choose_work_mode(seed, index, job_type, cfg):
    rng = rng_for(seed, "job", index, "work_mode")

    modes = cfg.get("work_mode_weights", DEFAULT_WORK_MODES)
    weights = modes.get(job_type, DEFAULT_WORK_MODES[job_type])

    return weighted_choice(rng, weights)


def choose_salary(seed, index, job_type, cfg):
    rng = rng_for(seed, "job", index, "salary")

    salary_ranges = cfg.get("salary_ranges_egp", DEFAULT_SALARY_RANGES)
    low, high = salary_ranges.get(
        job_type,
        DEFAULT_SALARY_RANGES[job_type],
    )

    minimum = rng.randint(int(low), int(high * 0.75))
    maximum = rng.randint(
        max(minimum + 1000, int(low * 1.15)),
        int(high),
    )

    return minimum, maximum


# ---------------------------------------------------------------------
# Assemble one job
# ---------------------------------------------------------------------

def build_job(index, registry, cfg, templates, template_data):
    """Generate one complete job document and its private metadata."""
    seed = cfg.get("seed", 42)

    job_id = f"job_{index:04d}"

    template_name, template, job_type = select_template(
        seed,
        index,
        templates,
        cfg,
    )

    title_rng = rng_for(seed, "job", index, "title")
    title = title_for_template(title_rng, template)

    location_rng = rng_for(seed, "job", index, "location")
    location = choose_city(location_rng, cfg)

    employer_rng = rng_for(seed, "job", index, "company")
    employers = template_data.get(
        "employers",
        ["Local Employer"],
    )
    company = employer_rng.choice(employers)

    salary = choose_salary(seed, index, job_type, cfg)

    # Make template-level text pools available to the text builder.
    text_cfg = dict(cfg)
    text_cfg["_template_data"] = template_data

    skills = build_skills(seed, index, template)

    summary, description, incomplete = build_job_text(
        seed,
        index,
        title,
        template,
        skills,
        company,
        salary,
        text_cfg,
    )

    demands, demand_metadata = build_demands(
        seed,
        index,
        template,
        job_type,
        registry,
        cfg,
    )

    workplace = build_workplace(
        seed,
        index,
        job_type,
        registry,
        cfg,
    )

    work_mode = choose_work_mode(
        seed,
        index,
        job_type,
        cfg,
    )

    status = choose_status(seed, index, cfg)

    job = {
        "job_id": job_id,
        "job_version": 1,
        "status": status,
        "job_type": job_type,
        "title": title,
        "summary": summary,
        "description": description,
        "work_mode": work_mode,
        "location": location,
        "skills": skills,
        "demands": demands,
        "workplace": workplace,
    }

    estimated_levels = template.get("est", [])

    if estimated_levels == "all":
        estimated_levels = list(template.get("levels", {}).keys())

    # Metadata is kept outside the M2 JSON document.
    # It is useful for inspecting generated data and tracing templates.
    ground_truth = {
        "job_id": job_id,
        "template_key": template_name,
        "esco_occupation": template.get("esco"),
        "onet": template.get("onet"),
        "intensity_levels": template.get("levels", {}),
        "estimated_levels": estimated_levels,
        "company": company,
        "salary_egp_monthly": {
            "minimum": salary[0],
            "maximum": salary[1],
        },
        "incomplete_posting": incomplete,
        **demand_metadata,
    }

    return job, ground_truth


# ---------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------

def generate_versions(seed, index, job, cfg):
    """
    Some jobs have multiple versions.

    The older version is archived and the latest version retains
    the status selected for the current posting.
    """
    stale_share = float(cfg.get("stale_version_share", 0.08))

    has_old_version = (
        rng_for(seed, "job", index, "stale_version").random()
        < stale_share
    )

    if not has_old_version:
        return [job]

    current = copy.deepcopy(job)
    current["job_version"] = 2

    old = copy.deepcopy(job)
    old["job_version"] = 1
    old["status"] = "archived"

    # The older posting is deliberately less complete.
    if old.get("description"):
        old["description"] = (
            "Earlier version of the vacancy. "
            + old["description"]
        )

    return [old, current]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def make_validator(root):
    schema_path = root / "schemas" / "job_m2.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Job schema not found: {schema_path}"
        )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft7Validator.check_schema(schema)

    return Draft7Validator(schema)


def validate_job(validator, job):
    """Validate one job and stop the generation run if it fails."""
    errors = sorted(
        validator.iter_errors(job),
        key=lambda error: list(error.path),
    )

    if errors:
        messages = [
            f"{'/'.join(map(str, error.path)) or '(root)'}: "
            f"{error.message}"
            for error in errors
        ]

        raise ValueError(
            f"Job {job['job_id']} version {job['job_version']} "
            "failed schema validation:\n  "
            + "\n  ".join(messages[:15])
        )

    # Additional consistency checks.
    demand_codes = [
        demand["dimension_code"]
        for demand in job.get("demands", [])
    ]

    if len(demand_codes) != len(set(demand_codes)):
        raise ValueError(
            f"{job['job_id']}: duplicate demand codes."
        )

    workplace_codes = [
        item["item_code"]
        for item in job.get("workplace", [])
    ]

    if len(workplace_codes) != len(set(workplace_codes)):
        raise ValueError(
            f"{job['job_id']}: duplicate workplace item codes."
        )

    for demand in job.get("demands", []):
        if demand["value_state"] == "unknown" and demand.get("value") is not None:
            raise ValueError(
                f"{job['job_id']}: unknown demand "
                f"{demand['dimension_code']} must have value=null."
            )

    for item in job.get("workplace", []):
        if item["value_state"] == "unknown" and item.get("value") is not None:
            raise ValueError(
                f"{job['job_id']}: unknown workplace item "
                f"{item['item_code']} must have value=null."
            )


# ---------------------------------------------------------------------
# Generate all jobs
# ---------------------------------------------------------------------

def generate_all(root, count=None):
    """
    Generate the requested number of jobs.

    Existing candidates sections in ground_truth.json and manifest.json
    are preserved.
    """
    root = Path(root)

    registry = load_yaml(root / "registry" / "registry.yaml")
    cfg = load_yaml(root / "registry" / "config.yaml")
    template_data = load_yaml(
        root / "registry" / "job_templates.yaml"
    )

    templates = get_templates(template_data)

    if count is None:
        count = int(cfg.get("n_jobs", 150))

    seed = cfg.get("seed", 42)

    validator = make_validator(root)

    output_dir = root / "out" / "jobs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_truth = {}
    by_category = {}
    by_status = {}
    by_template = {}
    version_count = 0

    for index in range(1, count + 1):
        job, truth = build_job(
            index,
            registry,
            cfg,
            templates,
            template_data,
        )

        versions = generate_versions(
            seed,
            index,
            job,
            cfg,
        )

        for version in versions:
            validate_job(validator, version)

            filename = (
                f"{version['job_id']}_v"
                f"{version['job_version']}.json"
            )

            write_json(output_dir / filename, version)
            version_count += 1

        all_truth[job["job_id"]] = truth

        category = job["job_type"]
        status = job["status"]
        template_name = truth["template_key"]

        by_category[category] = by_category.get(category, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        by_template[template_name] = (
            by_template.get(template_name, 0) + 1
        )

    # Keep the candidate data generated by candidates.py.
    merge_json_section(
        root / "out" / "ground_truth.json",
        "jobs",
        all_truth,
    )

    merge_json_section(
        root / "out" / "manifest.json",
        "jobs",
        {
            "seed": seed,
            "generator_version": cfg.get("generator_version", "0.1"),
            "count": count,
            "version_count": version_count,
            "schema": "job_m2.json",
            "by_category": by_category,
            "by_status": by_status,
            "by_template": by_template,
        },
    )

    print(f"Generated {count} jobs.")
    print(f"Generated {version_count} job versions.")
    print(f"Output directory: {output_dir}")
    print("Categories:", by_category)
    print("Statuses:", by_status)


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(
        sys.argv[1] if len(sys.argv) > 1 else "."
    )

    number_of_jobs = (
        int(sys.argv[2])
        if len(sys.argv) > 2
        else None
    )

    generate_all(project_root, number_of_jobs)
