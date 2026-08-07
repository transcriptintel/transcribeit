from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "check_module_size.py"
SPEC = importlib.util.spec_from_file_location("check_module_size", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class ModuleSizePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        self.original_repository = CHECKER.REPOSITORY
        self.original_policy = CHECKER.POLICY_PATH
        CHECKER.REPOSITORY = self.root
        CHECKER.POLICY_PATH = self.root / "policy.json"

    def tearDown(self) -> None:
        CHECKER.REPOSITORY = self.original_repository
        CHECKER.POLICY_PATH = self.original_policy
        self.temporary.cleanup()

    def write_module(self, relative: str, lines: int) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"// line {index}\n" for index in range(lines)))

    def write_policy(self, exceptions: list[dict[str, object]]) -> None:
        CHECKER.POLICY_PATH.write_text(
            json.dumps(
                {
                    "version": 1,
                    "exception_budget": len(exceptions),
                    "exceptions": exceptions,
                }
            )
        )

    @staticmethod
    def exception(path: str, ceiling: int) -> dict[str, object]:
        prefix = "Legacy " if ceiling > CHECKER.NEW_MODULE_LIMIT else "Reviewed "
        return {
            "path": path,
            "max_lines": ceiling,
            "rationale": prefix + "cohesive responsibility with an exact ratcheted ceiling.",
        }

    def test_new_large_module_and_exception_growth_fail(self) -> None:
        self.write_module("src/new.rs", 401)
        self.write_policy([])
        violations = CHECKER.evaluate(CHECKER.production_modules(), CHECKER.load_policy())
        self.assertTrue(any("400-line limit" in item for item in violations))

        self.write_policy([self.exception("src/new.rs", 400)])
        violations = CHECKER.evaluate(CHECKER.production_modules(), CHECKER.load_policy())
        self.assertTrue(any("above ceiling 400" in item for item in violations))

    def test_legacy_ceiling_passes_exactly_and_ratchets_after_shrink(self) -> None:
        self.write_module("src/legacy.rs", 500)
        self.write_policy([self.exception("src/legacy.rs", 500)])
        self.assertEqual(
            CHECKER.evaluate(CHECKER.production_modules(), CHECKER.load_policy()), []
        )
        self.write_module("src/legacy.rs", 499)
        violations = CHECKER.evaluate(CHECKER.production_modules(), CHECKER.load_policy())
        self.assertTrue(any("lower ceiling 500" in item for item in violations))

    def test_test_modules_are_not_production_policy_inputs(self) -> None:
        self.write_module("src/lib.rs", 10)
        self.write_module("src/pipeline/tests.rs", 900)
        self.write_policy([])
        modules = CHECKER.production_modules()
        self.assertIn("src/lib.rs", modules)
        self.assertNotIn("src/pipeline/tests.rs", modules)

    def test_legacy_exception_requires_explicit_rationale(self) -> None:
        self.write_module("src/legacy.rs", 500)
        exception = self.exception("src/legacy.rs", 500)
        exception["rationale"] = "Reviewed cohesive responsibility without the required marker."
        self.write_policy([exception])
        with self.assertRaisesRegex(CHECKER.PolicyError, "must be legacy"):
            CHECKER.load_policy()


class RepositoryValidatorTests(unittest.TestCase):
    def test_current_policy_and_skills_validate(self) -> None:
        commands = (
            ["python3", "-B", str(REPOSITORY / "scripts" / "check_module_size.py")],
            ["bun", "run", str(REPOSITORY / "scripts" / "validate_skills.ts")],
        )
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    command,
                    cwd=REPOSITORY,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
