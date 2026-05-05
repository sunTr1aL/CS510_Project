import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_edits import lexical_overlap, summarize
from experiments.clip_backend import winoground_decisions, write_diagnostics
from experiments.configs import EXPERIMENTS, get_run
from experiments.run_experiment import main
from scripts.build_structured_edit_jsonl import infer_edit_spans
from scripts.debug_loss_alignment import parse_indices, token_mask_from_ranges


class ExperimentScaffoldTests(unittest.TestCase):
    def test_registry_covers_tracker_runs(self):
        self.assertEqual(len(EXPERIMENTS), 45)
        self.assertEqual(get_run("r009").slot_count, 8)
        self.assertEqual(get_run("R011").slot_count, 8)
        self.assertEqual(get_run("R014").variant, "no_targeted_loss")
        self.assertEqual(get_run("R019").variant, "rrb_full")
        self.assertEqual(get_run("R027").seed, 3)
        self.assertEqual(get_run("R028").variant, "no_targeted_loss")
        self.assertEqual(get_run("R031").seed, 2)
        self.assertEqual(get_run("R032").variant, "no_targeted_loss")
        self.assertEqual(get_run("R033").stage, "train")
        self.assertEqual(get_run("R034").stage, "winoground_eval")
        self.assertEqual(get_run("R035").seed, 1)
        self.assertEqual(get_run("R039").seed, 5)
        self.assertEqual(get_run("R040").variant, "no_slot_residual_adapter")
        self.assertEqual(get_run("R041").stage, "slot_probe")
        self.assertEqual(get_run("R042").stage, "rop_basis")
        self.assertEqual(get_run("R043").variant, "rop_no_targeted_loss")
        self.assertEqual(get_run("R044").seed, 2)
        self.assertEqual(get_run("R045").stage, "flickr_retrieval")

    def test_audit_summary(self):
        summary = summarize(
            [
                {
                    "caption": "a cup left of a bowl",
                    "counterfactual": "a cup right of a bowl",
                    "filter_passed": True,
                    "filter_score_margin": 0.3,
                }
            ]
        )
        self.assertEqual(summary.count, 1)
        self.assertGreater(summary.mean_lexical_overlap, 0.5)
        self.assertEqual(summary.filter_keep_rate, 1.0)

    def test_cli_dry_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--run-id", "R009", "--dry-run", "--output-root", tmp])
            self.assertEqual(rc, 0)
            manifest = Path(tmp) / "R009" / "manifest.json"
            self.assertTrue(manifest.exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved"]["slot_count"], 8)
            self.assertEqual(payload["resolved"]["lambda_distill"], 0.0)
            self.assertEqual(payload["resolved"]["lambda_original"], 0.0)
            self.assertEqual(payload["resolved"]["lambda_targeted"], 0.0)
            self.assertEqual(payload["resolved"]["backend"], "adapter")

    def test_cli_token_backend_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--run-id",
                    "R009",
                    "--dry-run",
                    "--backend",
                    "token_bottleneck",
                    "--lambda-distill",
                    "0.2",
                    "--lambda-original",
                    "0.3",
                    "--lambda-targeted",
                    "0.4",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            manifest = Path(tmp) / "R009" / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved"]["backend"], "token_bottleneck")
            self.assertEqual(payload["resolved"]["lambda_distill"], 0.2)
            self.assertEqual(payload["resolved"]["lambda_original"], 0.3)
            self.assertEqual(payload["resolved"]["lambda_targeted"], 0.4)

    def test_cli_rrb_backend_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--run-id",
                    "R019",
                    "--dry-run",
                    "--backend",
                    "rrb",
                    "--lambda-anchor",
                    "5",
                    "--lambda-delta",
                    "0.01",
                    "--lambda-targeted",
                    "1",
                    "--residual-alpha",
                    "0.05",
                    "--anchor-ratio",
                    "0.5",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            manifest = Path(tmp) / "R019" / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved"]["backend"], "rrb")
            self.assertEqual(payload["resolved"]["lambda_anchor"], 5.0)
            self.assertEqual(payload["resolved"]["lambda_delta"], 0.01)
            self.assertEqual(payload["resolved"]["residual_alpha"], 0.05)

    def test_cli_m8_dry_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--run-id",
                    "R035",
                    "--dry-run",
                    "--backend",
                    "rrb",
                    "--lambda-anchor",
                    "20",
                    "--lambda-delta",
                    "0.01",
                    "--anchor-ratio",
                    "0.75",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads((Path(tmp) / "R035" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run"]["variant"], "no_targeted_loss")
            self.assertEqual(payload["resolved"]["lambda_anchor"], 20.0)
            self.assertEqual(payload["resolved"]["seed"], 1)

            rc = main(
                [
                    "--run-id",
                    "R040",
                    "--dry-run",
                    "--backend",
                    "rrb",
                    "--lambda-anchor",
                    "20",
                    "--lambda-delta",
                    "0.1",
                    "--anchor-ratio",
                    "0.75",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads((Path(tmp) / "R040" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run"]["variant"], "no_slot_residual_adapter")
            self.assertEqual(payload["resolved"]["lambda_delta"], 0.1)
            self.assertIsNone(payload["resolved"]["slot_count"])

    def test_cli_topvenue_dry_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--run-id",
                    "R043",
                    "--dry-run",
                    "--backend",
                    "rrb",
                    "--lambda-anchor",
                    "20",
                    "--lambda-delta",
                    "0.01",
                    "--lambda-rop",
                    "0.1",
                    "--rop-mode",
                    "penalty_project",
                    "--protected-basis",
                    "basis.pt",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads((Path(tmp) / "R043" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved"]["lambda_rop"], 0.1)
            self.assertEqual(payload["resolved"]["rop_mode"], "penalty_project")
            self.assertEqual(payload["resolved"]["protected_basis"], "basis.pt")

            rc = main(["--run-id", "R044", "--dry-run", "--output-name", "R044_seed3", "--seed", "3", "--output-root", tmp])
            self.assertEqual(rc, 0)
            payload = json.loads((Path(tmp) / "R044_seed3" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved"]["seed"], 3)
            self.assertEqual(payload["resolved"]["output_name"], "R044_seed3")

    def test_overlap_bounds(self):
        self.assertEqual(lexical_overlap("", ""), 1.0)
        self.assertEqual(lexical_overlap("cat", ""), 0.0)
        self.assertLessEqual(lexical_overlap("a b", "b c"), 1.0)

    def test_winoground_decisions(self):
        self.assertEqual(winoground_decisions(0.9, 0.1, 0.8, 0.2), (True, True, True))
        self.assertEqual(winoground_decisions(0.9, 0.1, 0.3, 0.8), (False, True, False))
        self.assertEqual(winoground_decisions(0.4, 0.7, 0.6, 0.2), (False, False, False))

    def test_loss_alignment_helpers(self):
        self.assertEqual(parse_indices("1, 3,5"), [1, 3, 5])
        ranges = {1: (0, 3, "cup"), 2: (4, 8, "left"), 3: (9, 11, "of")}
        mask = token_mask_from_ranges(ranges, 5, span="left", decoded_text="cup left of")
        self.assertEqual(mask, [False, False, True, False, False])

    def test_structured_edit_span_inference(self):
        edited, keep = infer_edit_spans("a cup left of a bowl", "a cup right of a bowl")
        self.assertEqual(edited, "left")
        self.assertTrue(keep in {"a cup", "of a bowl"})

    def test_diagnostics_from_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "R028"
            guardrail_dir = root / "guardrails" / "R028"
            out_dir = root / "R030"
            run_dir.mkdir(parents=True)
            guardrail_dir.mkdir(parents=True)
            checkpoint = run_dir / "adapter.pt"
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "checkpoint_eval_summary.json").write_text(
                json.dumps({"accuracy": 0.75, "total": 2}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "train_summary.json").write_text(
                json.dumps({"residual_relative_magnitude": 0.1}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "rrb_drift_summary.json").write_text(
                json.dumps(
                    {
                        "residual_relative_magnitude": {
                            "mean": 0.1,
                            "min": 0.05,
                            "max": 0.2,
                            "histogram": {"count": 2},
                        },
                        "z0_cosine": {
                            "mean": 0.99,
                            "min": 0.98,
                            "max": 1.0,
                            "histogram": {"count": 2},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (guardrail_dir / "guardrail_summary.json").write_text(
                json.dumps(
                    {
                        "coco_retrieval": {"image_to_text_r1": 0.7},
                        "imagenet_zeroshot": {"top1": 0.6},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "eval_records.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "index": 0,
                                "source": "src",
                                "edit_type": "rel",
                                "correct": False,
                                "margin": -0.2,
                                "positive_score": 0.1,
                                "best_negative_score": 0.3,
                                "predicted_index": 1,
                            }
                        ),
                        json.dumps(
                            {
                                "index": 1,
                                "source": "src",
                                "edit_type": "rel",
                                "correct": True,
                                "margin": 0.05,
                                "positive_score": 0.35,
                                "best_negative_score": 0.3,
                                "predicted_index": 0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = write_diagnostics(checkpoint, out_dir)
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["selected_run_id"], "R028")
            self.assertEqual(summary["records_analyzed"], 2)
            self.assertEqual(summary["failure_taxonomy"][0]["failures"], 1)
            self.assertTrue((out_dir / "diagnostics_report.md").exists())
            self.assertTrue((out_dir / "selected_failure_cases.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
