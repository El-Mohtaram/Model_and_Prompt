"""Build the stabilized m3_rules_only_reference_ready.ipynb and update m3_rules_only_reference.ipynb."""

from __future__ import annotations

import json
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11+",
        },
    }

    cells = []

    # Title
    cells.append(
        new_markdown_cell(
            "# M3 rules-only matching reference harness\n\n"
            "Python reference harness + C# gate spec. No database, network, cloud model, or LLM for decisions. Run top to bottom.\n"
        )
    )

    # Section 0: Header + config
    cells.append(new_markdown_cell("## 0. Header + config\n"))

    cells.append(
        new_code_cell(
            """from __future__ import annotations
import calendar
import hashlib
import hmac
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from jsonschema import Draft7Validator, Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field


def resolve_project_root() -> Path:
    \"\"\"Resolve SyntheticData project root reliably across notebook working directories.\"\"\"
    env_root = os.getenv("SYNTHETICDATA_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "registry" / "registry.yaml").exists():
            return candidate
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / "registry" / "registry.yaml").exists() and (base / "schemas").exists():
            return base
    fallback = Path(r"C:\\Users\\mohan\\PycharmProjects\\SyntheticData")
    return fallback if fallback.exists() else cwd


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TODAY = date(2026, 11, 2)
RANDOM_SEED = 42
PIPELINE_VERSION = "m3-rules-only-reference/1.0.0"
MAPPING_VERSION = "mapping-v0"
np.random.seed(RANDOM_SEED)


@dataclass(frozen=True)
class PolicyConfig:
    candidate_dir: str = "Final Data/candidates"
    job_dir: str = "Final Data/jobs"
    registry_path: str = "registry/registry.yaml"
    manifest_path: str = "out/manifest.json"
    ground_truth_path: str = "out/ground_truth.json"
    candidate_schema_path: str = "schemas/candidate_v0_2.json"
    job_schema_path: str = "schemas/job_m2.json"
    audit_store_dir: str = "audit_store"
    output_dir: str = "outputs"
    d02_sustained_threshold_min: int = 15
    d02_review_on_frequent: bool = True
    d02_lower_confidence: bool = False
    use_of03_for_d05: bool = False
    use_d06_rule: bool = False
    d06_use_max_of01_pc24: bool = True
    allow_d08_elevator: bool = True
    allow_d08_stairs: bool = True
    accommodation_catalog_enabled: bool = True
    borderline_fraction: float = 0.05
    too_many_unknown_essential: int = 2
    stale_months: int = 12
    confidence_medium_unknown_fraction: float = 0.20
    confidence_low_unknown_fraction: float = 0.50
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 42
    fairness_di_low: float = 0.80
    fairness_di_high: float = 1.25
    privacy_blocklist: tuple = ("capability_value", "diagnosis", "disability", "score", "WH_D")


CONFIG = PolicyConfig()


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def canonical_json(x: Any) -> str:
    def norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {
                unicodedata.normalize("NFC", str(k)): norm(z)
                for k, z in sorted(v.items(), key=lambda p: str(p[0]))
            }
        if isinstance(v, list):
            a = [norm(z) for z in v]
            if a and all(isinstance(z, str) for z in a):
                return sorted(a)
            if a and all(isinstance(z, dict) and "name" in z for z in a):
                return sorted(a, key=lambda z: z["name"])
            if a and all(isinstance(z, dict) and ("item_code" in z or "dimension_code" in z) for z in a):
                k = "item_code" if "item_code" in a[0] else "dimension_code"
                return sorted(a, key=lambda z: z[k])
            return a
        return unicodedata.normalize("NFC", v) if isinstance(v, str) else v

    return json.dumps(norm(x), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(cfg: PolicyConfig = CONFIG) -> str:
    return hashlib.sha256(canonical_json(asdict(cfg)).encode()).hexdigest()[:16]


def safe_log(event: str, fields: dict | None = None) -> dict:
    return {"event": re.sub(r"[^\\w.-]", "_", event), "fields": fields or {}}


def read_json(p: str | Path, default: Any) -> Any:
    rp = resolve_path(p)
    return json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else default


print(f"Project root resolved to: {PROJECT_ROOT}")
print(f"Reference date TODAY: {TODAY.isoformat()} | Config hash: {config_hash()}")
"""
        )
    )

    # Optional Kaggle inference diagnostics
    cells.append(
        new_markdown_cell(
            "### Optional Kaggle inference diagnostics\n\n"
            "الـ Kaggle client ده diagnostic/backend adapter فقط. مش بيتستخدم في gate، ranking، feasibility، batch، fairness، أو تفسير النتائج. "
            "مفيش اتصال تلقائي عند تشغيل Run All؛ شغّل خلايا الاختبار يدويًا بعد ضبط متغيرات البيئة. الرسائل تفضل role/content objects ومفيش ملفات المشروع بتتبعت تلقائيًا.\n"
        )
    )

    cells.append(
        new_code_cell(
            """from llm.kaggle_client import ask_llm, kaggle_health

KAGGLE_LLM_API_URL = os.getenv("KAGGLE_LLM_API_URL")
KAGGLE_LLM_CONFIGURED = bool(KAGGLE_LLM_API_URL and os.getenv("KAGGLE_LLM_API_TOKEN"))
print("Kaggle LLM configured (optional diagnostic only):", KAGGLE_LLM_CONFIGURED)

# Optional live health check. Uncomment and run manually after setting both environment variables.
# health = kaggle_health()
# print(health)

# Optional minimal model check. This is diagnostic only and sends only the literal test message below.
# answer = ask_llm([{"role": "user", "content": "Reply with exactly: KAGGLE_LLM_OK"}])
# print(answer)
"""
        )
    )

    # Section 1: Data loading
    cells.append(new_markdown_cell("## 1. Data loading"))

    cells.append(
        new_code_cell(
            """def golden_candidate_fixture() -> dict:
    return {
        "schema_version": "0.2",
        "candidate_id": "cand_0001",
        "job_category": "mobility",
        "disability": "AUDIT_ONLY",
        "skills": [],
        "cv_text": {},
        "capabilities": [
            {"item_code": "PC01", "value": 38, "unit": "kg", "method": "self_reported"},
            {"item_code": "PC02", "value": 21, "unit": "kg", "method": "self_reported"},
            {"item_code": "PC03", "value": 30, "unit": "kg", "method": "self_reported"},
            {"item_code": "PC04", "value": 31, "unit": "kg", "method": "self_reported"},
            {"item_code": "PC05", "value": 175, "unit": "m", "method": "self_reported"},
            {"item_code": "PC06", "value": 170, "unit": "kg", "method": "self_reported"},
            {"item_code": "PC07", "value": 180, "unit": "cm", "method": "self_reported"},
            {"item_code": "PC11", "value": 1060, "unit": "m", "method": "self_reported"},
            {"item_code": "PC12", "value": 7, "unit": "hours", "method": "self_reported"},
            {"item_code": "PC13", "value": 7, "unit": "flights", "method": "self_reported"},
            {"item_code": "PC14", "value": 21, "unit": "cm", "method": "self_reported"},
            {"item_code": "PC15", "value": 11, "unit": "1:x", "method": "self_reported"},
            {"item_code": "PC24", "value": 121, "unit": "minutes", "method": "self_reported"},
            {"item_code": "WH_D1_2", "score": None},
        ],
        "input_methods": ["keyboard", "voice", "touch"],
        "usable_formats": ["audio"],
        "communication_channels": ["spoken", "written"],
        "at_needs": ["screen_reader"],
        "preferences": {"remote": False, "location": "Cairo"},
    }


def golden_job_fixture() -> dict:
    return {
        "job_id": "job_0001",
        "job_version": 1,
        "status": "published",
        "job_type": "mobility",
        "title": "Installation Electrician",
        "work_mode": "hybrid",
        "location": "Alexandria",
        "demands": [
            {"dimension_code": "D01", "value_state": "known", "value": 20, "unit": "kg", "frequency_band": "never", "essential": False},
            {"dimension_code": "D02", "value_state": "known", "value": 120, "unit": "kg", "frequency_band": "frequent", "duration_min": 28, "essential": True},
            {"dimension_code": "D05", "value_state": "known", "value": 173, "unit": "min", "essential": False},
            {"dimension_code": "D06", "value_state": "unknown", "value": None, "unit": "min", "essential": False},
            {"dimension_code": "D07", "value_state": "known", "value": 610, "unit": "m", "essential": False},
            {"dimension_code": "D08", "value_state": "known", "value": True, "essential": True},
            {"dimension_code": "D09", "value_state": "known", "value": 1150, "unit": "mm", "essential": False},
            {"dimension_code": "D10", "value_state": "known", "value": 1450, "unit": "mm", "essential": False},
            {"dimension_code": "D11", "value_state": "known", "value": {"typical": 79.8, "peak": 92.2}, "unit": "dB(A)", "essential": False},
            {"dimension_code": "D12", "value_state": "known", "value": 450, "unit": "lux", "essential": False},
            {"dimension_code": "D13", "value_state": "known", "value": {"lowest": 21.3, "highest": 35.8}, "unit": "°C", "essential": False},
            {"dimension_code": "D15", "value_state": "known", "value": ["channel:spoken", "channel:written", "provision:captioning"], "essential": False},
            {"dimension_code": "D16", "value_state": "known", "value": ["stepwise", "verbal", "written"], "essential": False},
            {"dimension_code": "D17", "value_state": "known", "value": ["hybrid"], "essential": False},
        ],
        "workplace": [
            {"item_code": "E09", "value_state": "known", "value": True, "verification_level": "third_party_audited", "measured_at": "2024-10-16"},
            {"item_code": "E10", "value_state": "known", "value": True, "verification_level": "self_reported", "measured_at": "2025-08-20"},
            {"item_code": "E11", "value_state": "known", "value": 85, "unit": "m", "verification_level": "self_reported", "measured_at": "2024-05-01"},
            {"item_code": "E12", "value_state": "unknown", "value": None, "measured_at": None},
        ],
    }


CANDIDATE_SCHEMA_FALLBACK = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "candidate_id",
        "job_category",
        "disability",
        "skills",
        "cv_text",
        "capabilities",
        "input_methods",
        "usable_formats",
        "communication_channels",
        "at_needs",
        "preferences",
    ],
    "properties": {
        "schema_version": {"const": "0.2"},
        "candidate_id": {"type": "string"},
        "job_category": {"enum": ["mobility", "office", "both"]},
        "disability": {"type": "string"},
        "skills": {"type": "array"},
        "cv_text": {"type": "object"},
        "capabilities": {"type": "array"},
        "input_methods": {"type": "array"},
        "usable_formats": {"type": "array"},
        "communication_channels": {"type": "array"},
        "at_needs": {"type": "array"},
        "preferences": {"type": "object"},
    },
}

JOB_SCHEMA_FALLBACK = {
    "type": "object",
    "required": ["job_id", "job_version", "status", "job_type", "demands"],
}

LAST_LOAD_DIAGNOSTICS: dict[str, Any] = {}


def load_inputs(cfg: PolicyConfig = CONFIG) -> tuple[list[dict], list[dict], dict, dict, dict, dict]:
    \"\"\"Load candidates, jobs, registry, manifest, and schemas relative to PROJECT_ROOT with explicit source tracking.\"\"\"
    global LAST_LOAD_DIAGNOSTICS
    cand_dir = resolve_path(cfg.candidate_dir)
    job_dir = resolve_path(cfg.job_dir)
    reg_path = resolve_path(cfg.registry_path)

    cand_files = sorted(cand_dir.glob("*.json")) if cand_dir.exists() else []
    job_files = sorted(job_dir.glob("*.json")) if job_dir.exists() else []

    cs, cand_read_errors = [], []
    for p in cand_files:
        try:
            cs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            cand_read_errors.append({"file": p.name, "error": type(exc).__name__})

    js, job_read_errors = [], []
    for p in job_files:
        try:
            js.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            job_read_errors.append({"file": p.name, "error": type(exc).__name__})

    if cand_files and not cs:
        raise ValueError(f"Found {len(cand_files)} candidate files in {cand_dir}, but all failed to parse.")
    if job_files and not js:
        raise ValueError(f"Found {len(job_files)} job files in {job_dir}, but all failed to parse.")

    if cs and js:
        data_source = "real_project_data"
    elif not cs and not js:
        data_source = "golden_fixture_fallback"
        cs = [golden_candidate_fixture()]
        js = [golden_job_fixture()]
    else:
        data_source = "mixed_real_and_fallback"
        if not cs:
            cs = [golden_candidate_fixture()]
        if not js:
            js = [golden_job_fixture()]

    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    man = read_json(cfg.manifest_path, {"seed": 42, "count": 0})
    csch = read_json(cfg.candidate_schema_path, CANDIDATE_SCHEMA_FALLBACK)
    jsch = read_json(cfg.job_schema_path, JOB_SCHEMA_FALLBACK)

    LAST_LOAD_DIAGNOSTICS = {
        "data_source": data_source,
        "candidate_dir": str(cand_dir),
        "job_dir": str(job_dir),
        "candidate_files_found": len(cand_files),
        "job_files_found": len(job_files),
        "candidates_loaded": len(cs),
        "jobs_loaded": len(js),
        "published_jobs_loaded": sum(1 for j in js if j.get("status") == "published"),
        "candidate_read_errors": cand_read_errors,
        "job_read_errors": job_read_errors,
        "registry_loaded": bool(reg),
    }
    return cs, js, reg, man, csch, jsch
"""
        )
    )

    # Section 2: Privacy border
    cells.append(new_markdown_cell("## 2. Privacy border"))

    cells.append(
        new_code_cell(
            """class Capability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_code: str
    value: float | None = None
    score: int | None = None
    unit: str | None = None
    method: str | None = None
    value_state: Literal["known", "unknown"]
    used_by_gate: bool = True


class InternalCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["0.2"]
    candidate_id: str
    job_category: Literal["mobility", "office", "both"]
    skills: list[dict] = Field(default_factory=list)
    cv_text: dict[str, str | None] = Field(default_factory=dict)
    capabilities: list[Capability]
    input_methods: list[str]
    usable_formats: list[str]
    communication_channels: list[str]
    at_needs: list[str]
    preferences: dict


_VALIDATOR_CACHE: dict[int, Draft7Validator] = {}


def _get_draft7_validator(schema: dict) -> Draft7Validator:
    key = id(schema)
    if key not in _VALIDATOR_CACHE:
        _VALIDATOR_CACHE[key] = Draft7Validator(schema)
    return _VALIDATOR_CACHE[key]


def privacy_border(raw: dict, schema: dict, audit_dir: Path | str) -> tuple[dict, dict]:
    \"\"\"Validate raw candidate, store diagnosis solely in isolated audit store, and strip protected fields before gate.\"\"\"
    _get_draft7_validator(schema).validate(raw)
    resolved_audit_dir = resolve_path(audit_dir)
    audit = {"candidate_id": raw["candidate_id"], "disability": raw["disability"]}
    resolved_audit_dir.mkdir(parents=True, exist_ok=True)
    (resolved_audit_dir / f"{raw['candidate_id']}.json").write_text(
        json.dumps(audit, ensure_ascii=False), encoding="utf-8"
    )

    clean = {k: v for k, v in raw.items() if k != "disability"}
    caps = []
    for x in clean["capabilities"]:
        code = x["item_code"]
        if code.startswith("WH_D6_"):
            continue
        ignored = bool(re.fullmatch(r"H[1-3]", code))
        known = (not ignored and x.get("value", None) is not None) or (
            not ignored and x.get("score", None) is not None
        )
        caps.append({**x, "value_state": "known" if known else "unknown", "used_by_gate": not ignored})

    clean["capabilities"] = caps
    out = InternalCandidate.model_validate(clean).model_dump()
    assert "disability" not in out and "persona" not in out
    return out, audit
"""
        )
    )

    # Section 3: Registry + normalization
    cells.append(new_markdown_cell("## 3. Registry + normalization\n"))

    cells.append(
        new_code_cell(
            """def normalize_quantity(v: float | int | None, unit: str | None, target: str | None) -> float | None:
    if v is None:
        return None
    u = (unit or "").strip().lower()
    t = (target or "").strip().lower()
    if u == t or (u, t) in {("min", "minutes"), ("minutes", "min")}:
        return float(v)
    factors = {
        ("hours", "min"): 60,
        ("hour", "min"): 60,
        ("h", "min"): 60,
        ("seconds", "min"): 1 / 60,
        ("seconds", "minutes"): 1 / 60,
        ("cm", "mm"): 10,
        ("m", "mm"): 1000,
        ("hours", "minutes"): 60,
    }
    f = factors.get((u, t))
    return float(v) * f if f is not None else None


def capmap(c: dict) -> dict[str, dict]:
    return {x["item_code"]: x for x in c["capabilities"]}


def capval(c: dict, code: str, reg: dict) -> float | None:
    x = capmap(c).get(code)
    if not x or x["value_state"] != "known" or x.get("value") is None:
        return None
    meta = reg.get("candidate_items", {}).get(code, {})
    return normalize_quantity(x["value"], x.get("unit", meta.get("unit")), meta.get("unit"))


def findd(j: dict, code: str) -> dict | None:
    return next((x for x in j.get("demands", []) if x["dimension_code"] == code), None)


def findw(j: dict, code: str) -> dict | None:
    return next((x for x in j.get("workplace", []) if x["item_code"] == code), None)


def direction_sufficient(capacity: float, demand: float, hib: bool = True) -> bool:
    return capacity >= demand if hib else capacity <= demand
"""
        )
    )

    # Section 4 & 5: Mapping config, accommodation catalog, and comparators
    cells.append(new_markdown_cell("## 4. Mapping config + comparators\n\n## 5. Accommodation catalog"))

    cells.append(
        new_code_cell(
            """MAPPING = {
    "D01": {"status": "AMBIGUOUS", "items": ["PC01", "PC02", "PC03", "PC04", "PC19"]},
    "D02": {"status": "CLEAR", "items": ["PC06"]},
    "D03": {"status": "CLEAR", "items": ["input_methods"]},
    "D04": {"status": "CLEAR", "items": ["OF01"]},
    "D05": {"status": "CLEAR", "items": ["PC24"]},
    "D06": {"status": "AMBIGUOUS", "items": ["OF01", "PC24"]},
    "D07": {"status": "CLEAR", "items": ["PC11"]},
    "D08": {"status": "AMBIGUOUS", "items": ["PC13", "E09"]},
    **{f"D{i:02}": {"status": "NO_MATCH"} for i in range(9, 14)},
    "D14": {"status": "CLEAR", "items": ["usable_formats"]},
    "D15": {"status": "CLEAR", "items": ["communication_channels"]},
    "D16": {"status": "AMBIGUOUS"},
    "D17": {"status": "CLEAR", "items": ["preferences.remote"]},
}

ACCOMMODATION_CATALOG = [
    {"need_or_gap": "visual_to_audio", "resolution_id": "screen_reader_audio", "resolves_dimension": "D14", "candidate_need": "screen_reader"},
    {"need_or_gap": "communication", "resolution_id": "captioning", "resolves_dimension": "D15", "candidate_need": "captioning"},
    {"need_or_gap": "input_method", "resolution_id": "alternative_input", "resolves_dimension": "D03", "candidate_need": "switch_or_gaze"},
]


@dataclass(frozen=True)
class Comparison:
    status: str
    reason_code: str
    detail: dict
    resolution: str | None = None

    def __iter__(self):
        \"\"\"Support tuple unpacking (status, reason_code, resolution) as well as attribute access.\"\"\"
        yield self.status
        yield self.reason_code
        yield self.resolution


def R(status: str, reason_code: str, resolution: str | None = None, **detail: Any) -> Comparison:
    return Comparison(status=status, reason_code=reason_code, detail=detail, resolution=resolution)


def _catalog(code: str, key: str | None = None, cfg: PolicyConfig = CONFIG) -> str | None:
    if not cfg.accommodation_catalog_enabled:
        return None
    catalog_entries = [
        ("D14", "screen_reader", "screen_reader_audio"),
        ("D14", "visual_to_audio", "screen_reader_audio"),
        ("D15", "captioning", "captioning"),
        ("D15", "communication", "captioning"),
        ("D03", "input_method", "alternative_input"),
        ("D03", "switch_or_gaze", "alternative_input"),
    ]
    return next((rid for dim, k, rid in catalog_entries if dim == code and (key is None or k == key)), None)


def catalog(code: str, need: str | None = None, cfg: PolicyConfig = CONFIG) -> str | None:
    return _catalog(code, need, cfg)


def compare(code: str, c: dict, j: dict, reg: dict, cfg: PolicyConfig = CONFIG) -> Comparison:
    \"\"\"Authoritative comparator: unknown evidence stays Unknown, never Gap.\"\"\"
    d = findd(j, code)
    if not d:
        return R("Unknown", "demand_missing")
    if d.get("value_state") != "known" or d.get("value") is None:
        return R("Unknown", "demand_unknown")
    v = d["value"]

    def numeric(item: str) -> tuple[float | None, float | None]:
        cv = capval(c, item, reg)
        meta = reg.get("candidate_items", {}).get(item, {})
        dv = normalize_quantity(v, d.get("unit"), meta.get("unit"))
        return cv, dv

    if code == "D01":
        band = d.get("frequency_band")
        items = (
            ("PC01", "PC04")
            if band == "occasional"
            else (("PC02",) if band in ("frequent", "constant") else ("PC01", "PC02", "PC03", "PC04", "PC19"))
        )
        vals = [capval(c, x, reg) for x in items]
        vals = [x for x in vals if x is not None]
        if not vals:
            return R("Unknown", "candidate_value_missing")
        return R("Met" if max(vals) >= v else "Gap", "compared_capacity", compared=True)

    if code in ("D02", "D04", "D05", "D07"):
        item = {"D02": "PC06", "D04": "OF01", "D05": "PC24", "D07": "PC11"}[code]
        cv, dv = numeric(item)
        if code == "D05" and cfg.use_of03_for_d05:
            alt = capval(c, "OF03", reg)
            if cv is None and alt is None:
                return R("Unknown", "candidate_value_missing")
            best = max(x for x in (cv, alt) if x is not None)
            return R("Met" if (dv is not None and best >= dv) else "Gap", "compared_capacity", compared=True)
        if cv is None or dv is None:
            return R("Unknown", "candidate_value_missing")
        return R("Met" if cv >= dv else "Gap", "compared_capacity", compared=True)

    if code == "D03":
        if not c["input_methods"]:
            return R("Unknown", "candidate_value_missing")
        if set(c["input_methods"]) & set(v):
            return R("Met", "overlap", compared=True)
        res = _catalog(code, "input_method", cfg)
        if res:
            return R("Met with accommodation", "catalog_resolution", resolution=res, compared=True)
        return R("Gap", "no_overlap", compared=True)

    if code == "D08":
        if v is True:
            return R("Met", "step_free_route", compared=True)
        e = findw(j, "E09")
        if cfg.allow_d08_elevator and e and e.get("value_state") == "known" and e.get("value") is True:
            return R("Met with accommodation", "workplace_elevator", resolution="E09_elevator", compared=True)
        s = capval(c, "PC13", reg)
        if s is None:
            return R("Unknown", "candidate_value_missing")
        if cfg.allow_d08_stairs and s >= 1:
            return R("Met", "stairs_capacity", compared=True)
        return R("Gap", "route_steps_no_resolution", compared=True)

    if code == "D14":
        if set(c["usable_formats"]) & set(v):
            return R("Met", "overlap", compared=True)
        res = _catalog(code, "screen_reader", cfg)
        if res and "screen_reader" in c["at_needs"]:
            return R("Met with accommodation", "catalog_resolution", resolution=res, compared=True)
        return R("Gap", "no_overlap", compared=True)

    if code == "D15":
        ch = {x.split(":", 1)[1] for x in v if x.startswith("channel:")}
        if set(c["communication_channels"]) & ch:
            return R("Met", "overlap", compared=True)
        for p in sorted(x.split(":", 1)[1] for x in v if x.startswith("provision:")):
            res = _catalog(code, p, cfg)
            if res:
                return R("Met with accommodation", "catalog_resolution", resolution=res, compared=True)
        return R("Gap", "no_overlap", compared=True)

    if code == "D06":
        if not cfg.use_d06_rule:
            return R("Unknown", "mapping_unconfirmed")
        vals = [capval(c, x, reg) for x in ("OF01", "PC24")]
        vals = [x for x in vals if x is not None]
        if not vals:
            return R("Unknown", "candidate_value_missing")
        return R("Met" if max(vals) >= v else "Gap", "compared_capacity", compared=True)

    if code == "D16":
        return R("Unknown", "mapping_unconfirmed")

    if code in ("D09", "D10", "D11", "D12", "D13"):
        return R("Unknown", "no_candidate_counterpart")

    if code == "D17":
        opts = set(v)
        remote = bool(c["preferences"].get("remote"))
        mismatch = ("remote" not in opts) if remote else (opts != {"remote"})
        reason = "preference_mismatch" if mismatch else "preference_match"
        return R("Met", reason, informational=True, preference_status=reason)

    return R("Unknown", "mapping_unconfirmed")
"""
        )
    )

    # Section 6, 7, 8, 9: Aggregation, Confidence + Staleness, Review Flags, Output Builder
    cells.append(
        new_markdown_cell(
            "## 6. Job-level aggregation + final binary\n\n"
            "## 7. Confidence + staleness + missing_info\n\n"
            "## 8. Review flags\n\n"
            "## 9. Final output builder"
        )
    )

    cells.append(
        new_code_cell(
            """def subtract_months(reference_date: date, months: int) -> date:
    \"\"\"Subtract calendar months from reference_date with end-of-month clamping and leap-year support.\"\"\"
    if not isinstance(reference_date, date):
        raise TypeError("reference_date must be a datetime.date instance")
    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be an integer")
    if months < 0:
        raise ValueError("months must be non-negative")
    if months == 0:
        return reference_date
    total_months = (reference_date.year * 12 + (reference_date.month - 1)) - months
    target_year, target_month_zero = divmod(total_months, 12)
    target_month = target_month_zero + 1
    if target_year < 1:
        raise ValueError("Subtracted date precedes minimum valid calendar year")
    max_day = calendar.monthrange(target_year, target_month)[1]
    return date(target_year, target_month, min(reference_date.day, max_day))


def workplace_dates(j: dict, cfg: PolicyConfig = CONFIG, reference_date: date = TODAY) -> tuple[list[str], list[str]]:
    \"\"\"Canonical workplace measurement date classifier (stale vs undated).\"\"\"
    cutoff = subtract_months(reference_date, cfg.stale_months)
    stale: list[str] = []
    undated: list[str] = []
    for w in j.get("workplace", []):
        raw_dt = w.get("measured_at")
        dt: date | None = None
        if isinstance(raw_dt, str) and raw_dt.strip():
            try:
                dt = date.fromisoformat(raw_dt.strip())
            except ValueError:
                dt = None
        if dt is None:
            undated.append(w["item_code"])
        elif dt < cutoff:
            stale.append(w["item_code"])
    return sorted(stale), sorted(undated)


def stale_lists(j: dict, cfg: PolicyConfig = CONFIG, reference_date: date = TODAY) -> tuple[list[str], list[str]]:
    \"\"\"Delegate alias to canonical workplace_dates implementation.\"\"\"
    return workplace_dates(j, cfg=cfg, reference_date=reference_date)


def sanity(c: dict, reg: dict) -> list[str]:
    out = []
    for a, b in reg.get("constraints_le", []):
        x, y = capval(c, a, reg), capval(c, b, reg)
        if x is not None and y is not None and x > y:
            out.append(f"sanity:{a}_gt_{b}")
    return out


def registry_review_flags(c: dict, reg: dict) -> list[str]:
    flags = []
    for item in c["capabilities"]:
        meta = reg.get("candidate_items", {}).get(item["item_code"], {})
        if item.get("value_state") != "known" or item.get("value") is None or not meta.get("range"):
            continue
        lo, hi = meta["range"]
        if not lo <= item["value"] <= hi:
            flags.append("range_review:" + item["item_code"])
    return flags


OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "candidate_id",
        "job_id",
        "job_version",
        "dimension_results",
        "feasibility",
        "final_binary",
        "unverified_flag",
        "essential_unknown_count",
        "similarity",
        "confidence",
        "missing_info",
        "review_flags",
        "template_data",
        "employer_category",
        "model_version",
        "input_hash",
    ],
    "properties": {
        "candidate_id": {"type": "string"},
        "job_id": {"type": "string"},
        "job_version": {"type": "integer"},
        "dimension_results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["dimension", "status", "essential", "resolution", "reason_code"],
                "properties": {
                    "dimension": {"type": "string"},
                    "status": {"enum": ["Met", "Met with accommodation", "Gap", "Unknown"]},
                    "essential": {"type": "boolean"},
                    "resolution": {"type": ["string", "null"]},
                    "reason_code": {"type": "string"},
                },
            },
        },
        "feasibility": {"enum": ["Yes", "No", "Unverified"]},
        "final_binary": {"enum": [0, 1]},
        "unverified_flag": {"type": "boolean"},
        "essential_unknown_count": {"type": "integer"},
        "similarity": {},
        "confidence": {"type": "object"},
        "missing_info": {"type": "array"},
        "review_flags": {"type": "array"},
        "template_data": {"type": "object"},
        "employer_category": {"enum": ["fit", "fit_with_accommodation", "needs_review"]},
        "model_version": {"type": "string"},
        "input_hash": {"type": "string"},
    },
}

_OUTPUT_SCHEMA_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)


def build_output(c: dict, j: dict, reg: dict, similarity_provider: Any = None, cfg: PolicyConfig = CONFIG) -> dict:
    rows = []
    flags = []
    for d in sorted(j.get("demands", []), key=lambda x: x["dimension_code"]):
        s, reason, res = compare(d["dimension_code"], c, j, reg, cfg)
        rows.append(
            {
                "dimension": d["dimension_code"],
                "status": s,
                "essential": bool(d.get("essential", False)),
                "resolution": res,
                "reason_code": reason,
            }
        )
        if (
            d["dimension_code"] == "D02"
            and s == "Met"
            and cfg.d02_review_on_frequent
            and (
                d.get("frequency_band") in ("frequent", "constant")
                or (d.get("duration_min") or 0) > cfg.d02_sustained_threshold_min
            )
        ):
            flags.append("short_capacity_test_vs_sustained_demand")
        if (
            d["dimension_code"] in {"D01", "D02", "D04", "D05", "D06", "D07"}
            and s in {"Met", "Gap"}
            and isinstance(d.get("value"), (int, float))
            and d["value"] != 0
        ):
            cap_code = {"D01": "PC01", "D02": "PC06", "D04": "OF01", "D05": "PC24", "D06": "OF01", "D07": "PC11"}[
                d["dimension_code"]
            ]
            cv = capval(c, cap_code, reg)
            if cv is not None and abs(cv - d["value"]) / abs(d["value"]) <= cfg.borderline_fraction:
                flags.append("borderline:" + d["dimension_code"])

    ess = [r for r in rows if r["essential"]]
    feasibility = (
        "No"
        if any(r["status"] == "Gap" for r in ess)
        else ("Unverified" if any(r["status"] == "Unknown" for r in ess) else "Yes")
    )
    unknown = sum(r["status"] == "Unknown" for r in ess)
    ev = rows
    complete = sum(r["status"] != "Unknown" for r in ev) / len(ev) if ev else 1.0
    stale, undated = workplace_dates(j, cfg)

    if unknown >= cfg.too_many_unknown_essential:
        flags.append("too_many_unknown_essential")
    if stale:
        flags.append("stale_workplace_data")
    flags += sanity(c, reg) + registry_review_flags(c, reg)
    if rows and sum(r["status"] == "Unknown" for r in rows) / len(rows) >= cfg.confidence_low_unknown_fraction:
        flags.append("unknown_heavy")

    band = (
        "low"
        if complete < cfg.confidence_low_unknown_fraction
        else ("medium" if complete < cfg.confidence_medium_unknown_fraction or stale or undated else "high")
    )
    if cfg.d02_lower_confidence and "short_capacity_test_vs_sustained_demand" in flags:
        band = {"high": "medium", "medium": "low", "low": "low"}[band]

    sim = similarity_provider(c, j) if similarity_provider and feasibility != "No" else None
    missing = sorted(
        {r["dimension"] for r in rows if r["status"] == "Unknown"}
        | {w["item_code"] for w in j.get("workplace", []) if w.get("value_state") == "unknown"}
    )
    category = (
        "needs_review"
        if feasibility == "Unverified"
        else (
            "fit_with_accommodation"
            if any(r["essential"] and r["status"] == "Met with accommodation" for r in rows)
            else "fit"
        )
    )
    template = {
        k: [r["dimension"] for r in rows if r["status"] == s]
        for k, s in (("met", "Met"), ("accommodated", "Met with accommodation"), ("gaps", "Gap"), ("unknown", "Unknown"))
    }
    template["missing_info"] = missing
    template["catalog_resolutions_for_gaps"] = [
        x["resolution_id"]
        for x in ACCOMMODATION_CATALOG
        if any(r["dimension"] == x["resolves_dimension"] and r["status"] == "Gap" for r in rows)
    ]

    cfg_h = config_hash(cfg)
    inp = {
        "candidate": {k: v for k, v in c.items() if k not in ("cv_text", "skills", "job_category", "preferences")},
        "job": j,
        "preferences": c.get("preferences", {}),
        "versions": [PIPELINE_VERSION, cfg_h, reg.get("registry_version", "missing"), MAPPING_VERSION],
    }
    raw = canonical_json(inp).encode()
    key = os.environ.get("M3_INPUT_HMAC_KEY")
    digest = hmac.new(key.encode(), raw, hashlib.sha256).hexdigest() if key else hashlib.sha256(raw).hexdigest()

    employer = canonical_json({"employer_category": category, "template_data": template}).lower()
    if any(t.lower() in employer for t in cfg.privacy_blocklist):
        raise ValueError("Employer privacy blocklist violation")

    out = {
        "candidate_id": c["candidate_id"],
        "job_id": j["job_id"],
        "job_version": j["job_version"],
        "dimension_results": rows,
        "feasibility": feasibility,
        "final_binary": 0 if feasibility == "No" else 1,
        "unverified_flag": feasibility == "Unverified",
        "essential_unknown_count": unknown,
        "similarity": sim,
        "confidence": {
            "band": band,
            "data_completeness": round(complete, 6),
            "stale_items": stale,
            "undated_items": undated,
        },
        "missing_info": missing,
        "review_flags": sorted(set(flags)),
        "template_data": template,
        "employer_category": category,
        "model_version": f"{PIPELINE_VERSION}|{cfg_h}|{reg.get('registry_version', 'missing')}|{MAPPING_VERSION}",
        "input_hash": digest,
    }
    _OUTPUT_SCHEMA_VALIDATOR.validate(out)
    return out
"""
        )
    )

    # Section 10: Prompt-engineering slot
    cells.append(new_markdown_cell("## 10. PROMPT-ENGINEERING SLOT"))

    cells.append(
        new_code_cell(
            """# Section 10: empty prompt-engineering slot; no prompt/examples/schema authored.
PROMPT_CONFIG = {
    "enabled": False,
    "prompt_version": None,
    "model_name": None,
    "runtime": None,
    "seed": 42,
    "temperature": 0,
}
SYSTEM_PROMPT_TEMPLATE = "TODO: user supplies"
USER_PROMPT_TEMPLATE = "TODO: user supplies"
FEW_SHOT_EXAMPLES = []
DECODING_SCHEMA = None


def verify_prompt_slot_untouched() -> bool:
    assert PROMPT_CONFIG == {
        "enabled": False,
        "prompt_version": None,
        "model_name": None,
        "runtime": None,
        "seed": 42,
        "temperature": 0,
    }, "PROMPT_CONFIG must remain disabled and unmodified"
    assert SYSTEM_PROMPT_TEMPLATE == "TODO: user supplies", "SYSTEM_PROMPT_TEMPLATE must remain placeholder"
    assert USER_PROMPT_TEMPLATE == "TODO: user supplies", "USER_PROMPT_TEMPLATE must remain placeholder"
    assert FEW_SHOT_EXAMPLES == [], "FEW_SHOT_EXAMPLES must remain empty"
    assert DECODING_SCHEMA is None, "DECODING_SCHEMA must remain None"
    return True


@dataclass
class RedactionResult:
    redacted_text: str
    spans: list[dict]
    mapping: dict


LEXICON_PATH = resolve_path("config/sensitive_lexicon.txt")


def redact_before_llm(text: str) -> RedactionResult:
    terms = (
        [
            x.strip()
            for x in LEXICON_PATH.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        ]
        if LEXICON_PATH.exists()
        else []
    )
    out = text
    spans = []
    mapping = {}
    for i, term in enumerate(sorted(terms, key=len, reverse=True)):
        for m in list(re.finditer(re.escape(term), out, re.I))[::-1]:
            token = f"[REDACTED_{i}]"
            mapping[token] = m.group()
            spans.append({"start": m.start(), "end": m.end()})
            out = out[: m.start()] + token + out[m.end() :]
    return RedactionResult(out, sorted(spans, key=lambda x: x["start"]), mapping)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0 or k == 0:
        return 0.0
    lo, hi = 0.0, k / n

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k))

    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) > 1 - alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def recall_measurement(total: int, covered: int) -> dict:
    return {"recall": covered / total if total else None, "lower_bound": clopper_pearson_lower(covered, total), "n": total}


class ParsedSkillItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surface_text: str
    skill_label: str
    language: Literal["en", "ar", "ar_eg", "mixed", "franco"]
    polarity: Literal["affirmed", "negated", "hypothetical", "third_party", "unclear"]
    evidence_strength: Literal["explicit", "implicit_from_task", "unclear"]
    unresolved_reason: str | None = None
    importance_hint: str | None = None


class ParsedSkills(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    items: list[ParsedSkillItem]


class LocalLLMClient:
    def generate(self, prompt: str, schema: dict, seed: int) -> dict:
        raise NotImplementedError


def parse_free_text(text: str, mode: Literal["candidate_text", "job_text"], client: Any = None) -> ParsedSkills:
    if not PROMPT_CONFIG["enabled"]:
        return ParsedSkills(status="parser_not_configured", items=[])
    if client is None:
        raise ValueError("Implement a local-only adapter")
    return ParsedSkills.model_validate(
        client.generate(redact_before_llm(text).redacted_text, DECODING_SCHEMA, PROMPT_CONFIG["seed"])
    )


def validate_parsed_items(text: str, items: list[ParsedSkillItem]) -> dict:
    good = []
    affirmed = []
    bad = 0
    for x in items:
        pos = text.find(x.surface_text)
        if pos < 0:
            bad += 1
            continue
        row = x.model_dump() | {"start": pos, "end": pos + len(x.surface_text), "needs_candidate_confirmation": True}
        good.append(row)
        if x.polarity == "affirmed":
            affirmed.append(row)
    return {"items": good, "affirmed_skills": affirmed, "unsupported_extraction_rate": bad / len(items) if items else 0.0}


def prompt_lint(strings: list[str], path: Path = resolve_path("config/protected_terms.txt")) -> list[str]:
    terms = [x.strip().lower() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")] if path.exists() else []
    return sorted({t for s in strings for t in terms if t in s.lower()})


def prompt_version_hash() -> str:
    x = {
        "prompt": (SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE),
        "schema": DECODING_SCHEMA,
        "examples": FEW_SHOT_EXAMPLES,
        "model": PROMPT_CONFIG["model_name"],
        "chat_template_id": None,
        "runtime": PROMPT_CONFIG["runtime"],
        "seed": PROMPT_CONFIG["seed"],
    }
    return hashlib.sha256(canonical_json(x).encode()).hexdigest()


def append_prompt_registry() -> None:
    p = resolve_path(CONFIG.output_dir) / "prompt_registry.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "hash": prompt_version_hash(),
                "version": PROMPT_CONFIG["prompt_version"],
                "model": PROMPT_CONFIG["model_name"],
                "runtime": PROMPT_CONFIG["runtime"],
                "seed": PROMPT_CONFIG["seed"],
            }
        ]
    ).to_csv(p, index=False, mode="a", header=not p.exists())


def run_prompt_lint(strings: list[str]) -> bool:
    found = prompt_lint(strings)
    if found:
        raise AssertionError("Protected terms found")
    return True


def prompt_slot_tests(labelled_set_path: str | None = None, gold_path: str | None = None) -> dict:
    return {
        "redaction_recall": "SKIPPED" if not labelled_set_path else "READY",
        "parser_evaluation": "SKIPPED" if not gold_path else "READY",
        "prompt_tests": "SKIPPED" if not PROMPT_CONFIG["enabled"] else "READY",
    }


def evaluate_parser_stub(gold_path: str | None = None, predictions: Any = None) -> dict:
    if not gold_path or not resolve_path(gold_path).exists():
        return {
            "status": "SKIPPED",
            "schema_validity": "SKIPPED",
            "strict_skill_f1": "SKIPPED",
            "relaxed_skill_f1": "SKIPPED",
            "grounded_precision": "SKIPPED",
            "polarity_accuracy": "SKIPPED",
            "injection_attack_success": "SKIPPED",
            "latency_percentiles": "SKIPPED",
        }
    raise NotImplementedError("Supply a reviewed gold-set format before enabling evaluation")


verify_prompt_slot_untouched()
"""
        )
    )

    # Section 11: Test suite
    cells.append(new_markdown_cell("## 11. Test suite"))

    cells.append(
        new_code_cell(
            """def run_date_regression_tests() -> None:
    \"\"\"Regression tests for calendar month subtraction and workplace measurement staleness.\"\"\"
    # 1. Exact cutoff calculation for reference date 2026-11-02 minus 12 months
    cutoff = subtract_months(TODAY, CONFIG.stale_months)
    assert cutoff == date(2025, 11, 2), f"Expected 2025-11-02, got {cutoff}"

    # 2. Month subtraction examples (end-of-month clamping, leap years, year boundaries, zero months)
    assert subtract_months(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert subtract_months(date(2024, 3, 31), 1) == date(2024, 2, 29)
    assert subtract_months(date(2026, 1, 31), 1) == date(2025, 12, 31)
    assert subtract_months(date(2026, 11, 2), 0) == date(2026, 11, 2)

    # 3. Reject negative month counts
    try:
        subtract_months(TODAY, -1)
        raise AssertionError("Negative months must raise ValueError")
    except ValueError:
        pass

    # 4. Golden fixture stale & undated classification
    golden_stale, golden_undated = workplace_dates(golden_job_fixture())
    assert golden_stale == ["E09", "E10", "E11"]
    assert golden_undated == ["E12"]
    assert stale_lists(golden_job_fixture()) == (golden_stale, golden_undated)

    # 5. Boundary tests: 1 day before cutoff (stale), exact cutoff (not stale), 1 day after cutoff (not stale), invalid date (undated)
    boundary_job = {
        "workplace": [
            {"item_code": "E09", "value_state": "known", "value": True, "measured_at": "2025-11-01"},
            {"item_code": "E10", "value_state": "known", "value": True, "measured_at": "2025-11-02"},
            {"item_code": "E11", "value_state": "known", "value": 85, "measured_at": "2025-11-03"},
            {"item_code": "E12", "value_state": "unknown", "value": None, "measured_at": None},
            {"item_code": "E20", "value_state": "known", "value": True, "measured_at": "invalid-date"},
        ]
    }
    b_stale, b_undated = workplace_dates(boundary_job)
    assert b_stale == ["E09"]
    assert b_undated == ["E12", "E20"]

    # 6. Zero-month configuration
    cfg_zero = PolicyConfig(stale_months=0)
    z_stale, z_undated = workplace_dates(
        {
            "workplace": [
                {"item_code": "E09", "value_state": "known", "value": True, "measured_at": "2026-11-01"},
                {"item_code": "E10", "value_state": "known", "value": True, "measured_at": "2026-11-02"},
            ]
        },
        cfg=cfg_zero,
    )
    assert z_stale == ["E09"] and z_undated == []


def run_tests() -> None:
    \"\"\"Golden reference contract, determinism, date regression, and privacy checks.\"\"\"
    verify_prompt_slot_untouched()
    run_date_regression_tests()

    c, j, reg, manifest, cs, js = load_inputs()
    internal, _ = privacy_border(golden_candidate_fixture(), CANDIDATE_SCHEMA_FALLBACK, resolve_path(CONFIG.audit_store_dir))
    out = build_output(internal, golden_job_fixture(), reg)
    st = {x["dimension"]: x for x in out["dimension_results"]}
    assert st["D02"]["status"] == "Met" and "short_capacity_test_vs_sustained_demand" in out["review_flags"]
    assert st["D01"]["status"] == "Met" and st["D05"]["status"] == "Gap" and st["D06"]["status"] == "Unknown"
    assert st["D07"]["status"] == "Met" and st["D08"]["status"] == "Met"
    assert all(st[f"D{i:02}"]["status"] == "Unknown" for i in range(9, 14))
    assert st["D15"]["status"] == "Met" and st["D16"]["status"] == "Unknown"
    assert out["feasibility"] == "Yes" and out["final_binary"] == 1 and not out["unverified_flag"]
    assert st["D17"]["reason_code"] == "preference_mismatch"
    assert out["confidence"]["stale_items"] == ["E09", "E10", "E11"]
    assert out["confidence"]["undated_items"] == ["E12"]
    assert {"D06", "D09", "D10", "D11", "D12", "D13", "D16", "E12"} <= set(out["missing_info"])
    assert build_output(internal, golden_job_fixture(), reg) == out
    assert parse_free_text("x", "candidate_text").status == "parser_not_configured"
    assert normalize_quantity(1, "hours", "min") == normalize_quantity(60, "min", "min") == normalize_quantity(3600, "seconds", "min")

    missing = {**internal, "capabilities": [x for x in internal["capabilities"] if x["item_code"] != "PC06"]}
    unverified = build_output(missing, golden_job_fixture(), reg)
    assert unverified["feasibility"] == "Unverified" and unverified["final_binary"] == 1 and unverified["unverified_flag"]

    for item, meta in reg.get("candidate_items", {}).items():
        if meta.get("hib", True) is False:
            assert direction_sufficient(2, 3, False) and not direction_sufficient(4, 3, False)

    # Nonessential gaps and Unknown never alter job-level feasibility.
    assert build_output(internal, golden_job_fixture(), reg)["final_binary"] == 1
    stale_copy = json.loads(json.dumps(golden_job_fixture()))
    for w in stale_copy.get("workplace", []):
        w["measured_at"] = "2026-10-01"
    fresh = build_output(internal, stale_copy, reg)
    assert fresh["final_binary"] == out["final_binary"] and [x["status"] for x in fresh["dimension_results"]] == [
        x["status"] for x in out["dimension_results"]
    ]
    assert direction_sufficient(4, 3, True) and not direction_sufficient(2, 3, True)
    assert out["final_binary"] == (0 if out["feasibility"] == "No" else 1)
    assert all(st[f"D{i:02}"]["status"] != "Gap" for i in range(9, 14))
    assert "disability" not in internal and "persona" not in internal
    print("Golden, date regression, determinism, contract, privacy and core invariant checks passed.")
"""
        )
    )

    # Section 12: Batch run
    cells.append(new_markdown_cell("## 12. Batch run\n"))

    cells.append(
        new_code_cell(
            """LAST_BATCH_DIAGNOSTICS: dict[str, Any] = {}


def run_batch(
    candidates: list[dict],
    jobs: list[dict],
    reg: dict,
    sample_pairs: int | None = None,
    cfg: PolicyConfig = CONFIG,
    cand_schema: dict | None = None,
    job_schema: dict | None = None,
) -> pd.DataFrame:
    \"\"\"Execute deterministic matching batch across published jobs with explicit failure diagnostics.\"\"\"
    global LAST_BATCH_DIAGNOSTICS
    csch = cand_schema or globals().get("CANDIDATE_SCHEMA") or read_json(cfg.candidate_schema_path, CANDIDATE_SCHEMA_FALLBACK)
    jsch = job_schema or globals().get("JOB_SCHEMA") or read_json(cfg.job_schema_path, JOB_SCHEMA_FALLBACK)
    audit_dir = resolve_path(cfg.audit_store_dir)
    out_dir = resolve_path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    job_validator = _get_draft7_validator(jsch)
    failure_categories: Counter = Counter()
    failure_records: list[dict] = []

    published_jobs = [j for j in jobs if isinstance(j, dict) and j.get("status") == "published"]
    skipped_unpublished_jobs = len(jobs) - len(published_jobs)
    eligible_published_pairs = len(candidates) * len(published_jobs)

    # Pre-validate and sanitize candidates once at privacy border
    valid_candidates: list[tuple[str, dict]] = []
    invalid_candidate_count = 0
    for idx, c in enumerate(candidates):
        cid = str(c.get("candidate_id", f"candidate_index_{idx}")) if isinstance(c, dict) else f"candidate_index_{idx}"
        try:
            clean, _ = privacy_border(c, csch, audit_dir)
            valid_candidates.append((cid, clean))
        except Exception as exc:
            invalid_candidate_count += 1
            cat = f"candidate_validation:{type(exc).__name__}"
            failure_categories[cat] += len(published_jobs)
            failure_records.append(
                {
                    "scope": "candidate",
                    "candidate_id": cid,
                    "job_id": None,
                    "category": cat,
                }
            )

    # Pre-validate published jobs once
    valid_jobs: list[tuple[str, dict]] = []
    invalid_job_count = 0
    for idx, j in enumerate(published_jobs):
        jid = str(j.get("job_id", f"job_index_{idx}")) if isinstance(j, dict) else f"job_index_{idx}"
        try:
            job_validator.validate(j)
            valid_jobs.append((jid, j))
        except Exception as exc:
            invalid_job_count += 1
            cat = f"job_validation:{type(exc).__name__}"
            # Count pairs that were not already counted under invalid candidates
            failure_categories[cat] += len(valid_candidates)
            failure_records.append(
                {
                    "scope": "job",
                    "candidate_id": None,
                    "job_id": jid,
                    "category": cat,
                }
            )

    valid_pair_inputs = [(cid, clean, jid, j) for cid, clean in valid_candidates for jid, j in valid_jobs]
    skipped_by_sample_limit = 0
    if sample_pairs is not None and sample_pairs < len(valid_pair_inputs):
        skipped_by_sample_limit = len(valid_pair_inputs) - sample_pairs
        valid_pair_inputs = valid_pair_inputs[:sample_pairs]

    rows: list[dict] = []
    pair_execution_failures = 0
    for cid, clean, jid, j in valid_pair_inputs:
        try:
            rows.append(build_output(clean, j, reg, cfg=cfg))
        except Exception as exc:
            pair_execution_failures += 1
            cat = f"pair_execution:{type(exc).__name__}"
            failure_categories[cat] += 1
            failure_records.append(
                {
                    "scope": "pair",
                    "candidate_id": cid,
                    "job_id": jid,
                    "category": cat,
                }
            )

    failed_pairs = sum(failure_categories.values())
    status = "success" if rows and failed_pairs == 0 else ("partial_success" if rows else "empty_or_no_results")

    LAST_BATCH_DIAGNOSTICS = {
        "status": status,
        "total_candidates": len(candidates),
        "valid_candidates": len(valid_candidates),
        "invalid_candidates": invalid_candidate_count,
        "total_jobs": len(jobs),
        "published_jobs": len(published_jobs),
        "skipped_unpublished_jobs": skipped_unpublished_jobs,
        "valid_published_jobs": len(valid_jobs),
        "invalid_published_jobs": invalid_job_count,
        "eligible_published_pairs": eligible_published_pairs,
        "processed_pairs": len(rows),
        "skipped_pairs": skipped_by_sample_limit,
        "failed_pairs": failed_pairs,
        "failure_categories": dict(failure_categories),
        "failure_records_sample": failure_records[:50],
    }

    frame = pd.DataFrame(rows)
    frame.attrs["batch_diagnostics"] = LAST_BATCH_DIAGNOSTICS

    # Save full CSV with JSON-serialized complex columns for unambiguous tabular inspection
    csv_frame = frame.copy()
    for col in ("dimension_results", "confidence", "missing_info", "review_flags", "template_data"):
        if col in csv_frame.columns:
            csv_frame[col] = csv_frame[col].map(lambda v: json.dumps(v, ensure_ascii=False))
    csv_frame.to_csv(out_dir / "m3_results.csv", index=False, encoding="utf-8")

    # Save readable sample JSON
    sample_records = rows[:10]
    (out_dir / "sample.json").write_text(json.dumps(sample_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame
"""
        )
    )

    # Section 13: Fairness / invariance harness
    cells.append(new_markdown_cell("## 13. Fairness / invariance harness\n\n"))

    cells.append(
        new_code_cell(
            """def _as_rows(value: Any) -> list[dict]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value if isinstance(value, list) else []


def fairness_report(frame: pd.DataFrame, replicates: int = CONFIG.bootstrap_replicates, cfg: PolicyConfig = CONFIG) -> dict:
    \"\"\"Authoritative fairness & hard-constraint violation report using isolated audit group labels.\"\"\"
    audit_groups_path = resolve_path(cfg.audit_store_dir) / "audit_groups.json"
    if audit_groups_path.exists():
        gt = read_json(audit_groups_path, {"candidates": {}}).get("candidates", {})
        grouping_source = str(audit_groups_path)
    else:
        gt = read_json(cfg.ground_truth_path, {"candidates": {}}).get("candidates", {})
        grouping_source = str(resolve_path(cfg.ground_truth_path))

    audit = pd.DataFrame([{"candidate_id": k, "persona": v.get("persona")} for k, v in gt.items()])
    joined = frame.merge(audit, on="candidate_id", how="inner") if (len(frame) and len(audit)) else pd.DataFrame()

    metrics = []
    rng = np.random.default_rng(cfg.bootstrap_seed)
    yes_count = 0
    violations = 0
    unknown_dims = 0
    total_dims = 0

    for _, row in frame.iterrows():
        dims = _as_rows(row.get("dimension_results"))
        total_dims += len(dims)
        unknown_dims += sum(x.get("status") == "Unknown" for x in dims)
        is_yes = row.get("feasibility") == "Yes"
        yes_count += int(is_yes)
        violations += int(
            is_yes
            and any(
                d.get("essential")
                and d.get("status") == "Gap"
                and d.get("reason_code")
                not in ("demand_unknown", "candidate_value_missing", "no_candidate_counterpart", "mapping_unconfirmed")
                for d in dims
            )
        )

    if len(joined):
        overall = float(joined.final_binary.mean())
        for group, g in joined.groupby("persona"):
            vals = g.final_binary.to_numpy(dtype=float)
            rate = float(vals.mean())
            boots = (
                [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(replicates)]
                if len(vals)
                else []
            )
            lows, highs = np.quantile(boots, [0.025, 0.975]).tolist() if boots else [None, None]
            n_u = n_d = 0
            for _, row in g.iterrows():
                dims = _as_rows(row.get("dimension_results"))
                n_d += len(dims)
                n_u += sum(x.get("status") == "Unknown" for x in dims)
            di_val = round(rate / overall, 6) if overall else None
            metrics.append(
                {
                    "group": group,
                    "n": int(len(g)),
                    "selection_rate": round(rate, 6),
                    "selection_rate_ci95": [round(lows, 6) if lows is not None else None, round(highs, 6) if highs is not None else None],
                    "SPD": round(rate - overall, 6),
                    "DI": di_val,
                    "within_alarm_band": bool(di_val is not None and cfg.fairness_di_low <= di_val <= cfg.fairness_di_high),
                    "unknown_rate": round(float(g.unverified_flag.mean()), 6),
                    "unknown_dimension_rate": round(n_u / n_d, 6) if n_d else None,
                }
            )

    return {
        "overall_selection_rate": round(float(joined.final_binary.mean()), 6) if len(joined) else None,
        "evaluated_pairs": int(len(frame)),
        "audit_matched_pairs": int(len(joined)),
        "grouping_source": grouping_source,
        "group_metrics": metrics,
        "hard_constraint_violation_rate": violations / yes_count if yes_count else 0.0,
        "hard_constraint_violations": int(violations),
        "unknown_dimension_rate": round(unknown_dims / total_dims, 6) if total_dims else None,
        "alarm_range": [cfg.fairness_di_low, cfg.fairness_di_high],
        "warning": (
            "Initial engineering alarms, not legal safe harbor or certification. "
            "Group differences may reflect real capability-vs-demand differences in synthetic profiles; "
            "report them transparently, never hide them or loosen hard constraints."
        ),
    }


def invariance_checks(internal: dict, job: dict, registry: dict) -> bool:
    \"\"\"Verify counterfactual invariance across non-capability text, protected attributes, and preferences.\"\"\"
    base = build_output(internal, job, registry)
    changed = {
        **internal,
        "cv_text": {"experience": "Changed assistive technology brand, awards, and employment-gap text."},
        "skills": [],
    }
    other = build_output(changed, job, registry)
    assert (
        base["dimension_results"] == other["dimension_results"]
        and base["final_binary"] == other["final_binary"]
        and base["input_hash"] == other["input_hash"]
    )
    for field in ("persona", "past_occupation", "name", "gender", "age"):
        rejected = False
        try:
            InternalCandidate.model_validate({**internal, field: "audit_only"})
        except Exception:
            rejected = True
        assert rejected, f"{field} unexpectedly accepted"
    pref_changed = {**internal, "preferences": {"remote": True, "location": "Cairo"}}
    assert build_output(pref_changed, job, registry)["final_binary"] == base["final_binary"]
    return True
"""
        )
    )

    # Section 14: gate_spec export
    cells.append(
        new_markdown_cell(
            "## 14. gate_spec export\n\n"
            "**C# gate spec (mapping-v0).** Remove raw disability before gate processing; reject extra internal fields. "
            "Drop WH_D6; H1-H3 and null values are Unknown/ignored. Read unit/range/hib from registry; only known conversions are permitted. "
            "Range and sanity checks produce review flags only.\n\n"
            "D01: never/unspecified uses max known PC01/02/03/04/19; occasional uses PC01/PC04; frequent/constant uses PC02. "
            "D02 compares PC06 with D02, marks sustained-test review for frequent/constant or duration above 15 minutes, and keeps Met status. "
            "D03 input method overlap, then accommodation catalog, else Gap; missing candidate list Unknown. "
            "D04 OF01, D05 PC24 (OF03 optional/off), D06 Unknown/mapping_unconfirmed by default (optional max(OF01,PC24) rule off), D07 PC11. "
            "D08 step-free Met; else E09 elevator gives accommodated Met; PC13 >=1 gives Met; known zero without elevator Gap; missing PC13 Unknown. "
            "D09-D13 Unknown/no_candidate_counterpart. D14 format overlap then catalog else Gap. D15 channel overlap then catalog provision else Gap. "
            "D16 Unknown/mapping_unconfirmed. D17 informational preference only. Unknown never Gap. Skills/job_category never gate.\n\n"
            "Essential aggregation: any Gap means No; else any Unknown means Unverified; else Yes. Binary is 0 iff No; always emit unverified flag. "
            "1 is not a recommendation and 0 never blocks applying. Stale is older than 12 calendar months before TODAY (cutoff 2025-11-02 for TODAY 2026-11-02); "
            "undated/invalid is undated; both affect confidence only. Completeness = known statuses / evaluated statuses. No cost field. "
            "Golden cand_0001 x job_0001 v1: PC06 170 >= D02 120 => Met plus sustained-test flag; D05 nonessential Gap; D06 Unknown; D08 Met; "
            "D09-D13 Unknown; D15 Met; D16 Unknown; Yes, binary 1, unverified false; stale E09/E10/E11; undated E12; similarity null.\n"
        )
    )

    # Section 15: Open assumptions and questions for M1/M2/M4/TA
    cells.append(
        new_markdown_cell(
            "## 15. Open assumptions and questions for M1/M2/M4/TA\n\n"
            "- M1: WH_D1..D5 semantics are outstanding; unused. Registry ranges are assumptions. Confirm hib=false directions and scales.\n"
            "- M2: Confirm D01, D06, D16, D08, D02 sustained-demand caveat, D05 PC24/OF03, and E-code meanings. D09-D13 stay Unknown.\n"
            "- M4: Preference rule ambiguity: removing preferences invariance conflicts with D17 informational comparison and golden preference mismatch; "
            "this reference keeps D17 informational (status `Met` with `preference_match`/`preference_mismatch` reason code) and excludes it from feasibility and `final_binary`.\n"
            "- TA/privacy reviewer: confirm audit-store access and retention. Audit records never enter gate/hash/log output.\n"
            "- Responsible-AI reviewer: binary collapses Unverified into 1; flag is mandatory. Group differences can reflect real capability-demand differences and must be reported, "
            "not hidden or fixed by loosening constraints. DI 0.80-1.25 and zero hard-constraint violations are initial engineering alarms, not legal safe harbor.\n"
            "- Assumption: accommodation rows are mock and must be confirmed as available. Confidence bands use configured completeness cutoffs; stale/undated lowers confidence only. "
            "Generator verification weights are not evidence weights.\n"
            "- Assumption: only published jobs enter the batch. Recall measurement on a labelled redaction set and parser evaluation remain skipped until the corresponding paths are supplied; "
            "this applies only to the isolated prompt slot.\n"
        )
    )

    # Section 16: Full Pipeline Execution, Artifact Generation & Visible Results
    cells.append(
        new_markdown_cell(
            "## 16. Pipeline Execution, Golden Demonstration & Results Inspection\n\n"
            "Executes the complete deterministic pipeline top-to-bottom, saves all required artifacts to `outputs/`, and displays the structured reports."
        )
    )

    cells.append(
        new_code_cell(
            """from IPython.display import Markdown, display

# 1. Load inputs and validate schemas
RAW_CANDIDATES, JOBS, REGISTRY, MANIFEST, CANDIDATE_SCHEMA, JOB_SCHEMA = load_inputs()
INTERNAL_CANDIDATES, AUDIT_RECORDS = [], {}
for raw in RAW_CANDIDATES:
    internal, audit = privacy_border(raw, CANDIDATE_SCHEMA, resolve_path(CONFIG.audit_store_dir))
    INTERNAL_CANDIDATES.append(internal)
    AUDIT_RECORDS[internal["candidate_id"]] = audit

assert not any("disability" in x or "persona" in x for x in INTERNAL_CANDIDATES)

# 2. Run golden, regression, and invariance checks
run_tests()
if any(j.get("job_id") == "job_0001" for j in JOBS):
    invariance_checks(
        INTERNAL_CANDIDATES[0],
        next(j for j in JOBS if j.get("job_id") == "job_0001"),
        REGISTRY,
    )
    print("Counterfactual invariance checks passed on dataset job_0001.")

# 3. Copy only audit grouping labels into the isolated audit store for fairness reporting
_ground_truth = read_json(CONFIG.ground_truth_path, {"candidates": {}}).get("candidates", {})
audit_groups_file = resolve_path(CONFIG.audit_store_dir) / "audit_groups.json"
audit_groups_file.parent.mkdir(parents=True, exist_ok=True)
audit_groups_file.write_text(
    json.dumps(
        {
            "candidates": {
                k: {"persona": v.get("persona"), "past_occupation": v.get("past_occupation")}
                for k, v in _ground_truth.items()
            }
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# 4. Execute full deterministic batch across all candidates and published jobs
RESULTS = run_batch(RAW_CANDIDATES, JOBS, REGISTRY, cand_schema=CANDIDATE_SCHEMA, job_schema=JOB_SCHEMA)
FAIRNESS_REPORT = fairness_report(RESULTS)

# 5. Build staleness report across published jobs and golden fixture
STALE_CUTOFF_DATE = subtract_months(TODAY, CONFIG.stale_months)
published_jobs_list = [j for j in JOBS if j.get("status") == "published"]
job_staleness_rows = []
stale_item_counter: Counter = Counter()
undated_item_counter: Counter = Counter()

for j in published_jobs_list:
    s_items, u_items = workplace_dates(j, CONFIG)
    stale_item_counter.update(s_items)
    undated_item_counter.update(u_items)
    job_staleness_rows.append(
        {
            "job_id": j["job_id"],
            "stale_count": len(s_items),
            "undated_count": len(u_items),
            "stale_items": s_items,
            "undated_items": u_items,
        }
    )

golden_stale, golden_undated = workplace_dates(golden_job_fixture(), CONFIG)
STALE_DATA_REPORT = {
    "reference_date": TODAY.isoformat(),
    "stale_threshold_months": CONFIG.stale_months,
    "calculated_cutoff_date": STALE_CUTOFF_DATE.isoformat(),
    "golden_fixture_check": {
        "job_id": "job_0001 (golden_job_fixture)",
        "stale_items": golden_stale,
        "undated_items": golden_undated,
    },
    "published_jobs_evaluated": len(published_jobs_list),
    "jobs_with_stale_items": sum(1 for r in job_staleness_rows if r["stale_count"] > 0),
    "jobs_with_undated_items": sum(1 for r in job_staleness_rows if r["undated_count"] > 0),
    "stale_item_counts_by_code": dict(sorted(stale_item_counter.items())),
    "undated_item_counts_by_code": dict(sorted(undated_item_counter.items())),
    "per_job_staleness": job_staleness_rows,
}

# 6. Build run summary and save all required output files
feas_counts = RESULTS["feasibility"].value_counts().to_dict() if len(RESULTS) else {}
bin_counts = {str(k): int(v) for k, v in RESULTS["final_binary"].value_counts().to_dict().items()} if len(RESULTS) else {}
conf_counts = (
    RESULTS["confidence"].map(lambda x: x["band"]).value_counts().to_dict()
    if len(RESULTS)
    else {}
)
unknown_by_dim = Counter(
    d["dimension"]
    for items in (RESULTS["dimension_results"] if len(RESULTS) else [])
    for d in items
    if d["status"] == "Unknown"
)

RUN_SUMMARY = {
    "pipeline_version": PIPELINE_VERSION,
    "mapping_version": MAPPING_VERSION,
    "config_hash": config_hash(CONFIG),
    "reference_date": TODAY.isoformat(),
    "stale_cutoff_date": STALE_CUTOFF_DATE.isoformat(),
    "prompt_slot_enabled": PROMPT_CONFIG["enabled"],
    "load_diagnostics": LAST_LOAD_DIAGNOSTICS,
    "batch_diagnostics": LAST_BATCH_DIAGNOSTICS,
    "output_rows": int(len(RESULTS)),
    "distributions": {
        "feasibility": {k: int(v) for k, v in feas_counts.items()},
        "final_binary": bin_counts,
        "unverified_flag": {str(k): int(v) for k, v in RESULTS["unverified_flag"].value_counts().to_dict().items()} if len(RESULTS) else {},
        "confidence_band": {k: int(v) for k, v in conf_counts.items()},
        "unknown_by_dimension": dict(sorted(unknown_by_dim.items())),
    },
}

out_dir = resolve_path(CONFIG.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "run_summary.json").write_text(json.dumps(RUN_SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
(out_dir / "fairness_report.json").write_text(json.dumps(FAIRNESS_REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
(out_dir / "stale_data_report.json").write_text(json.dumps(STALE_DATA_REPORT, ensure_ascii=False, indent=2), encoding="utf-8")

print("Batch execution and artifact generation complete.")
"""
        )
    )

    # Display cell 1: Execution summary, Feasibility, Final Binary, Confidence, Unknowns, Staleness
    cells.append(
        new_code_cell(
            """# --- 1. Execution Summary ---
display(Markdown("### 1. Execution Summary"))
summary_df = pd.DataFrame(
    [
        {"Metric": "Execution status", "Value": LAST_BATCH_DIAGNOSTICS["status"]},
        {"Metric": "Data source used", "Value": LAST_LOAD_DIAGNOSTICS["data_source"]},
        {"Metric": "Candidates loaded", "Value": LAST_BATCH_DIAGNOSTICS["total_candidates"]},
        {"Metric": "Jobs loaded (total)", "Value": LAST_BATCH_DIAGNOSTICS["total_jobs"]},
        {"Metric": "Published jobs", "Value": LAST_BATCH_DIAGNOSTICS["published_jobs"]},
        {"Metric": "Skipped unpublished jobs", "Value": LAST_BATCH_DIAGNOSTICS["skipped_unpublished_jobs"]},
        {"Metric": "Eligible published candidate/job pairs", "Value": LAST_BATCH_DIAGNOSTICS["eligible_published_pairs"]},
        {"Metric": "Successfully processed pairs", "Value": LAST_BATCH_DIAGNOSTICS["processed_pairs"]},
        {"Metric": "Failed pairs", "Value": LAST_BATCH_DIAGNOSTICS["failed_pairs"]},
        {"Metric": "Skipped pairs (sample limit)", "Value": LAST_BATCH_DIAGNOSTICS["skipped_pairs"]},
        {"Metric": "Output rows generated", "Value": len(RESULTS)},
    ]
)
display(summary_df)

# --- 2. Feasibility Distribution ---
display(Markdown("### 2. Feasibility Distribution"))
feas_series = RESULTS["feasibility"].value_counts().reindex(["Yes", "No", "Unverified"], fill_value=0)
feas_df = pd.DataFrame(
    {
        "feasibility": feas_series.index,
        "count": feas_series.values,
        "percentage": (feas_series.values / len(RESULTS) * 100.0).round(2) if len(RESULTS) else 0.0,
    }
)
display(feas_df)

# --- 3. Final Binary Distribution ---
display(Markdown("### 3. Final Binary Distribution"))
bin_series = RESULTS["final_binary"].value_counts().reindex([1, 0], fill_value=0)
bin_df = pd.DataFrame(
    {
        "final_binary": bin_series.index,
        "meaning": ["Feasible or Unverified (never auto-rejected)", "Infeasible (essential Gap with no resolution)"],
        "count": bin_series.values,
        "percentage": (bin_series.values / len(RESULTS) * 100.0).round(2) if len(RESULTS) else 0.0,
    }
)
display(bin_df)

# --- 4. Confidence Distribution ---
display(Markdown("### 4. Confidence Band Distribution"))
conf_series = RESULTS["confidence"].map(lambda x: x["band"]).value_counts().reindex(["high", "medium", "low"], fill_value=0)
conf_df = pd.DataFrame(
    {
        "confidence_band": conf_series.index,
        "count": conf_series.values,
        "percentage": (conf_series.values / len(RESULTS) * 100.0).round(2) if len(RESULTS) else 0.0,
    }
)
display(conf_df)

# --- 5. Unknown Dimension Analysis ---
display(Markdown("### 5. Unknown Dimension Analysis"))
all_dims = [f"D{i:02}" for i in range(1, 18)]
total_eval_by_dim = Counter(d["dimension"] for items in RESULTS["dimension_results"] for d in items)
unknown_rows = []
for dim in all_dims:
    u_cnt = unknown_by_dim.get(dim, 0)
    tot_cnt = total_eval_by_dim.get(dim, 0)
    unknown_rows.append(
        {
            "dimension": dim,
            "unknown_count": u_cnt,
            "evaluated_count": tot_cnt,
            "unknown_pct_when_present": round(u_cnt / tot_cnt * 100.0, 2) if tot_cnt else 0.0,
        }
    )
display(pd.DataFrame(unknown_rows))

# --- 6. Staleness Report ---
display(Markdown("### 6. Workplace Staleness Report"))
print(f"Reference Date (TODAY): {TODAY.isoformat()} | Configured Stale Threshold: {CONFIG.stale_months} calendar months | Cutoff Date: {STALE_CUTOFF_DATE.isoformat()}")
print(f"Golden Fixture Stale Items: {golden_stale} | Golden Fixture Undated Items: {golden_undated}")
stale_summary_df = pd.DataFrame(
    [
        {
            "item_code": code,
            "stale_count_across_published_jobs": stale_item_counter.get(code, 0),
            "undated_count_across_published_jobs": undated_item_counter.get(code, 0),
        }
        for code in sorted(set(stale_item_counter) | set(undated_item_counter))
    ]
)
display(stale_summary_df)
"""
        )
    )

    # Display cell 2: Fairness Report, Actual Result Records, Golden Fixture Demonstration, Artifact Paths
    cells.append(
        new_code_cell(
            """# --- 7. Fairness Report ---
display(Markdown("### 7. Fairness Report (Audit Store Persona Groups)"))
print(f"Overall selection rate (final_binary == 1): {FAIRNESS_REPORT['overall_selection_rate']}")
print(f"Hard-constraint violation rate: {FAIRNESS_REPORT['hard_constraint_violation_rate']} (violations: {FAIRNESS_REPORT['hard_constraint_violations']})")
print(f"Overall unknown dimension rate: {FAIRNESS_REPORT['unknown_dimension_rate']}")
print(f"Engineering alarm range for DI: {FAIRNESS_REPORT['alarm_range']}")
print(f"Notice: {FAIRNESS_REPORT['warning']}")
fairness_df = pd.DataFrame(FAIRNESS_REPORT["group_metrics"])
display(fairness_df)

# --- 8. Actual Result Records (RESULTS DataFrame) ---
display(Markdown("### 8. Actual Result Records (`RESULTS` DataFrame Sample & Dimension Breakdown)"))
display_cols = [
    "candidate_id",
    "job_id",
    "feasibility",
    "final_binary",
    "unverified_flag",
    "essential_unknown_count",
    "confidence",
    "review_flags",
    "missing_info",
    "employer_category",
]
display(RESULTS[display_cols].head(20))

display(Markdown("#### Inspectable Dimension-Level Results for First Result Record (`cand_0001` x `job_0001`)"))
first_record = RESULTS.iloc[0]
print(f"Candidate: {first_record['candidate_id']} | Job: {first_record['job_id']} | Feasibility: {first_record['feasibility']} | Hash: {first_record['input_hash']}")
display(pd.DataFrame(first_record["dimension_results"]))

# --- 9. Golden Fixture Demonstration ---
display(Markdown("### 9. Separate Golden Fixture Demonstration (`golden_candidate_fixture` x `golden_job_fixture`)"))
golden_internal, _ = privacy_border(golden_candidate_fixture(), CANDIDATE_SCHEMA_FALLBACK, resolve_path(CONFIG.audit_store_dir))
GOLDEN_RESULT = build_output(golden_internal, golden_job_fixture(), REGISTRY)
golden_overview = {k: v for k, v in GOLDEN_RESULT.items() if k != "dimension_results"}
print("Golden Fixture Summary Object:")
print(json.dumps(golden_overview, ensure_ascii=False, indent=2))
display(Markdown("#### Golden Fixture Dimension-Level Statuses"))
display(pd.DataFrame(GOLDEN_RESULT["dimension_results"]))

# --- 10. Generated Output Artifacts Verification ---
display(Markdown("### 10. Generated Output Files"))
expected_artifacts = [
    out_dir / "m3_results.csv",
    out_dir / "sample.json",
    out_dir / "run_summary.json",
    out_dir / "fairness_report.json",
    out_dir / "stale_data_report.json",
]
artifact_rows = []
for p in expected_artifacts:
    exists = p.exists()
    size_bytes = p.stat().st_size if exists else 0
    artifact_rows.append({"File": p.name, "Exists": exists, "Size (bytes)": size_bytes, "Full Path": str(p.resolve())})
    print(f"Artifact saved: {p.resolve()} ({size_bytes:,} bytes)")
display(pd.DataFrame(artifact_rows))
"""
        )
    )

    nb.cells = cells
    return nb


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    nb = build_notebook()
    ready_path = root / "m3_rules_only_reference_ready.ipynb"
    orig_path = root / "m3_rules_only_reference.ipynb"

    nbformat.validate(nb)
    nbformat.write(nb, ready_path)
    nbformat.write(nb, orig_path)
    print(f"Wrote {ready_path} and {orig_path} successfully.")


if __name__ == "__main__":
    main()
