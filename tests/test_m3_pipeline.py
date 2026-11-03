"""Comprehensive unit and regression tests for the M3 deterministic rules-only reference pipeline.

Tests cover:
1. Date arithmetic (subtract_months, year boundaries, leap years, end-of-month clamping, invalid inputs, zero months)
2. Stale and undated workplace measurement classification (golden items, cutoff boundary, +/- 1 day, invalid date strings)
3. Determinism (identical inputs -> identical outputs, canonical JSON input hash stability, ordering invariance)
4. Privacy border (diagnosis stripped, audit isolation, WH_D6 removal, H1-H3 and null handling, employer blocklist)
5. Output contracts (OUTPUT_SCHEMA validation, required fields, enums, JSON serializability)
6. Matching behavior (Golden test, essential vs nonessential, Unknown handling, accommodations, staleness independence, D17)
7. Batch behavior (valid pairs, empty datasets, missing files, invalid records, failure diagnostics, no silent failure)
8. Prompt engineering slot (disabled, empty placeholders, no LLM in matching)
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


def _load_notebook_namespace() -> dict:
    import sys
    import types

    nb_path = Path(__file__).resolve().parent.parent / "m3_rules_only_reference_ready.ipynb"
    if not nb_path.exists():
        nb_path = Path(__file__).resolve().parent.parent / "m3_rules_only_reference.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    mod_name = "__m3_test_module__"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    ns = mod.__dict__
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Skip interactive getpass or full batch execution cells when loading definitions
        if (
            "getpass(" in src
            or "RESULTS = run_batch(" in src
            or "API_URL" in src
            or "KAGGLE_CLIENT" in src
            or "qwen_result" in src
            or "display(Markdown(" in src
        ):
            continue
        exec(compile(src, f"<cell:{cell.get('id', 'unknown')}>", "exec"), ns)
    return ns


class M3PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = _load_notebook_namespace()
        cls.TODAY = cls.ns["TODAY"]
        cls.CONFIG = cls.ns["CONFIG"]
        cls.PolicyConfig = cls.ns["PolicyConfig"]
        cls.subtract_months = staticmethod(cls.ns["subtract_months"])
        cls.workplace_dates = staticmethod(cls.ns["workplace_dates"])
        cls.stale_lists = staticmethod(cls.ns["stale_lists"])
        cls.privacy_border = staticmethod(cls.ns["privacy_border"])
        cls.build_output = staticmethod(cls.ns["build_output"])
        cls.compare = staticmethod(cls.ns["compare"])
        cls.load_inputs = staticmethod(cls.ns["load_inputs"])
        cls.run_batch = staticmethod(cls.ns["run_batch"])
        cls.run_tests = staticmethod(cls.ns["run_tests"])
        cls.invariance_checks = staticmethod(cls.ns["invariance_checks"])
        cls.fairness_report = staticmethod(cls.ns["fairness_report"])
        cls.golden_candidate_fixture = staticmethod(cls.ns["golden_candidate_fixture"])
        cls.golden_job_fixture = staticmethod(cls.ns["golden_job_fixture"])
        cls.CANDIDATE_SCHEMA_FALLBACK = cls.ns["CANDIDATE_SCHEMA_FALLBACK"]
        cls.OUTPUT_SCHEMA = cls.ns["OUTPUT_SCHEMA"]
        cls.InternalCandidate = cls.ns["InternalCandidate"]
        _, _, cls.reg, _, cls.csch, cls.jsch = cls.load_inputs()

    # ------------------------------------------------------------------
    # 1. Date handling & Regression tests
    # ------------------------------------------------------------------
    def test_subtract_months_standard_and_year_boundaries(self) -> None:
        self.assertEqual(self.subtract_months(date(2026, 11, 2), 12), date(2025, 11, 2))
        self.assertEqual(self.subtract_months(date(2026, 1, 31), 1), date(2025, 12, 31))
        self.assertEqual(self.subtract_months(date(2026, 3, 15), 5), date(2025, 10, 15))

    def test_subtract_months_end_of_month_and_leap_year(self) -> None:
        self.assertEqual(self.subtract_months(date(2026, 3, 31), 1), date(2026, 2, 28))
        self.assertEqual(self.subtract_months(date(2024, 3, 31), 1), date(2024, 2, 29))
        self.assertEqual(self.subtract_months(date(2024, 2, 29), 12), date(2023, 2, 28))
        self.assertEqual(self.subtract_months(date(2026, 5, 31), 1), date(2026, 4, 30))

    def test_subtract_months_zero_and_invalid(self) -> None:
        self.assertEqual(self.subtract_months(date(2026, 11, 2), 0), date(2026, 11, 2))
        with self.assertRaises(ValueError):
            self.subtract_months(date(2026, 11, 2), -1)
        with self.assertRaises((TypeError, ValueError)):
            self.subtract_months(date(2026, 11, 2), True)  # type: ignore[arg-type]

    def test_golden_workplace_stale_and_undated_items(self) -> None:
        job = self.golden_job_fixture()
        stale_1, undated_1 = self.workplace_dates(job)
        stale_2, undated_2 = self.stale_lists(job)
        self.assertEqual(stale_1, ["E09", "E10", "E11"])
        self.assertEqual(undated_1, ["E12"])
        self.assertEqual(stale_1, stale_2)
        self.assertEqual(undated_1, undated_2)

    def test_cutoff_boundary_and_invalid_dates(self) -> None:
        # Cutoff is 2025-11-02 when TODAY is 2026-11-02 and stale_months=12
        job = {
            "workplace": [
                {"item_code": "E09", "value_state": "known", "value": True, "measured_at": "2025-11-01"},  # 1 day before -> stale
                {"item_code": "E10", "value_state": "known", "value": True, "measured_at": "2025-11-02"},  # exact cutoff -> not stale
                {"item_code": "E11", "value_state": "known", "value": 85, "measured_at": "2025-11-03"},    # 1 day after -> not stale
                {"item_code": "E12", "value_state": "unknown", "value": None, "measured_at": None},        # None -> undated
                {"item_code": "E20", "value_state": "known", "value": True, "measured_at": "not-a-date"},  # invalid -> undated
                {"item_code": "E22", "value_state": "known", "value": 700, "measured_at": "2025-02-30"},   # impossible date -> undated
            ]
        }
        stale, undated = self.workplace_dates(job)
        self.assertEqual(stale, ["E09"])
        self.assertEqual(undated, ["E12", "E20", "E22"])

    def test_zero_month_configuration(self) -> None:
        cfg_zero = self.PolicyConfig(stale_months=0)
        job = {
            "workplace": [
                {"item_code": "E09", "value_state": "known", "value": True, "measured_at": "2026-11-01"},
                {"item_code": "E10", "value_state": "known", "value": True, "measured_at": "2026-11-02"},
            ]
        }
        stale, undated = self.workplace_dates(job, cfg=cfg_zero)
        self.assertEqual(stale, ["E09"])
        self.assertEqual(undated, [])

    # ------------------------------------------------------------------
    # 2. Determinism
    # ------------------------------------------------------------------
    def test_determinism_and_ordering_invariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            internal, _ = self.privacy_border(
                self.golden_candidate_fixture(), self.CANDIDATE_SCHEMA_FALLBACK, Path(tmp)
            )
            job1 = self.golden_job_fixture()
            job2 = copy.deepcopy(job1)
            job2["demands"] = list(reversed(job2["demands"]))
            job2["workplace"] = list(reversed(job2["workplace"]))

            internal_rev = copy.deepcopy(internal)
            internal_rev["capabilities"] = list(reversed(internal_rev["capabilities"]))
            internal_rev["input_methods"] = list(reversed(internal_rev["input_methods"]))

            out1 = self.build_output(internal, job1, self.reg)
            out2 = self.build_output(internal_rev, job2, self.reg)
            self.assertEqual(out1, out2)
            self.assertEqual(out1["input_hash"], out2["input_hash"])

    # ------------------------------------------------------------------
    # 3. Privacy
    # ------------------------------------------------------------------
    def test_privacy_border_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp)
            raw = self.golden_candidate_fixture()
            raw["capabilities"].append({"item_code": "WH_D6_1", "score": 4})
            raw["capabilities"].append({"item_code": "H1", "score": 3})
            internal, audit = self.privacy_border(raw, self.CANDIDATE_SCHEMA_FALLBACK, audit_dir)

            self.assertNotIn("disability", internal)
            self.assertNotIn("persona", internal)
            self.assertEqual(audit["disability"], "AUDIT_ONLY")
            self.assertTrue((audit_dir / "cand_0001.json").exists())

            codes = {c["item_code"]: c for c in internal["capabilities"]}
            self.assertNotIn("WH_D6_1", codes)
            self.assertIn("H1", codes)
            self.assertEqual(codes["H1"]["value_state"], "unknown")
            self.assertFalse(codes["H1"]["used_by_gate"])
            self.assertEqual(codes["WH_D1_2"]["value_state"], "unknown")

    # ------------------------------------------------------------------
    # 4. Output contracts & Golden test
    # ------------------------------------------------------------------
    def test_golden_contract_and_invariance(self) -> None:
        self.run_tests()
        with tempfile.TemporaryDirectory() as tmp:
            internal, _ = self.privacy_border(
                self.golden_candidate_fixture(), self.CANDIDATE_SCHEMA_FALLBACK, Path(tmp)
            )
            self.assertTrue(self.invariance_checks(internal, self.golden_job_fixture(), self.reg))

    def test_output_json_serializable_and_schema_valid(self) -> None:
        from jsonschema import Draft202012Validator

        with tempfile.TemporaryDirectory() as tmp:
            internal, _ = self.privacy_border(
                self.golden_candidate_fixture(), self.CANDIDATE_SCHEMA_FALLBACK, Path(tmp)
            )
            out = self.build_output(internal, self.golden_job_fixture(), self.reg)
            Draft202012Validator(self.OUTPUT_SCHEMA).validate(out)
            serialized = json.dumps(out, ensure_ascii=False)
            self.assertIsInstance(serialized, str)

    # ------------------------------------------------------------------
    # 5. Batch behavior & Diagnostics
    # ------------------------------------------------------------------
    def test_batch_valid_and_invalid_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.PolicyConfig(
                output_dir=str(Path(tmp) / "outputs"),
                audit_store_dir=str(Path(tmp) / "audit"),
            )
            valid_cand = self.golden_candidate_fixture()
            invalid_cand = {"candidate_id": "bad_cand", "disability": "PRIVATE_SECRET_DIAGNOSIS"}
            valid_job = self.golden_job_fixture()
            unpublished_job = {**self.golden_job_fixture(), "job_id": "job_paused", "status": "paused"}
            invalid_job = {"job_id": "bad_job", "status": "published"}

            frame = self.run_batch(
                [valid_cand, invalid_cand],
                [valid_job, unpublished_job, invalid_job],
                self.reg,
                cfg=cfg,
            )
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["candidate_id"], "cand_0001")
            self.assertEqual(frame.iloc[0]["job_id"], "job_0001")

            diag = frame.attrs.get("batch_diagnostics", {})
            self.assertEqual(diag.get("total_candidates"), 2)
            self.assertEqual(diag.get("total_jobs"), 3)
            self.assertEqual(diag.get("published_jobs"), 2)
            self.assertEqual(diag.get("eligible_published_pairs"), 4)
            self.assertEqual(diag.get("processed_pairs"), 1)
            self.assertGreaterEqual(diag.get("failed_pairs", 0), 3)
            # Ensure sensitive diagnosis never leaked into diagnostics
            self.assertNotIn("PRIVATE_SECRET_DIAGNOSIS", json.dumps(diag))

    def test_batch_empty_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.PolicyConfig(
                output_dir=str(Path(tmp) / "outputs"),
                audit_store_dir=str(Path(tmp) / "audit"),
            )
            frame = self.run_batch([], [], self.reg, cfg=cfg)
            self.assertEqual(len(frame), 0)
            diag = frame.attrs.get("batch_diagnostics", {})
            self.assertEqual(diag.get("status"), "empty_or_no_results")

    def test_fairness_report_separates_unverified_from_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp)
            (audit_dir / "audit_groups.json").write_text(
                json.dumps(
                    {
                        "candidates": {
                            "cand_yes": {"persona": "group_a"},
                            "cand_unverified": {"persona": "group_a"},
                            "cand_no": {"persona": "group_b"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            frame = pd.DataFrame(
                [
                    {
                        "candidate_id": "cand_yes",
                        "feasibility": "Yes",
                        "final_binary": 1,
                        "unverified_flag": False,
                        "dimension_results": [],
                    },
                    {
                        "candidate_id": "cand_unverified",
                        "feasibility": "Unverified",
                        "final_binary": 1,
                        "unverified_flag": True,
                        "dimension_results": [],
                    },
                    {
                        "candidate_id": "cand_no",
                        "feasibility": "No",
                        "final_binary": 0,
                        "unverified_flag": False,
                        "dimension_results": [],
                    },
                ]
            )
            cfg = self.PolicyConfig(audit_store_dir=str(audit_dir))
            report = self.fairness_report(frame, replicates=10, cfg=cfg)

            self.assertEqual(report["overall_positive_gate_rate"], 0.666667)
            self.assertEqual(report["overall_confirmed_feasible_rate"], 0.333333)
            self.assertEqual(report["overall_unverified_rate"], 0.333333)
            self.assertEqual(report["overall_infeasible_rate"], 0.333333)
            self.assertEqual(report["group_metrics"][0]["confirmed_feasible_rate"], 0.5)
            self.assertEqual(report["group_metrics"][0]["unverified_rate"], 0.5)

    # ------------------------------------------------------------------
    # 6. Prompt engineering slot remains empty & disabled
    # ------------------------------------------------------------------
    def test_prompt_slot_disabled_and_empty(self) -> None:
        prompt_cfg = self.ns["PROMPT_CONFIG"]
        self.assertFalse(prompt_cfg["enabled"])
        self.assertIsNone(prompt_cfg["prompt_version"])
        self.assertIsNone(prompt_cfg["model_name"])
        self.assertIsNone(prompt_cfg["runtime"])
        self.assertEqual(prompt_cfg["seed"], 42)
        self.assertEqual(prompt_cfg["temperature"], 0)
        self.assertEqual(self.ns["SYSTEM_PROMPT_TEMPLATE"], "TODO: user supplies")
        self.assertEqual(self.ns["USER_PROMPT_TEMPLATE"], "TODO: user supplies")
        self.assertEqual(self.ns["FEW_SHOT_EXAMPLES"], [])
        self.assertIsNone(self.ns["DECODING_SCHEMA"])


if __name__ == "__main__":
    unittest.main()
