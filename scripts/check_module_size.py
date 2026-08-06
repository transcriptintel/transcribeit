#!/usr/bin/env python3
"""Enforce TranscribeIt's reviewed Rust production-module size ratchet."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPOSITORY = Path(__file__).resolve().parents[1]
POLICY_PATH = REPOSITORY / "scripts" / "module_size_policy.json"
TARGET_LINES = 300
NEW_MODULE_LIMIT = 400
LEGACY_CEILING_LIMIT = 900
MAX_EXCEPTION_BUDGET = 14
REPORT_COUNT = 20


class PolicyError(ValueError):
    """The checked-in module-size policy is malformed."""


@dataclass(frozen=True)
class ExceptionRule:
    max_lines: int
    rationale: str


def count_lines(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))


def production_modules() -> dict[str, int]:
    modules: dict[str, int] = {}
    source_root = REPOSITORY / "src"
    for path in sorted(source_root.rglob("*.rs")):
        relative = path.relative_to(REPOSITORY)
        if path.is_symlink():
            raise PolicyError(f"symlink is not allowed under src: {relative.as_posix()}")
        if path.name == "tests.rs" or "tests" in relative.parts:
            continue
        modules[relative.as_posix()] = count_lines(path)
    return modules


def load_policy() -> dict[str, ExceptionRule]:
    try:
        raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyError(f"cannot read {POLICY_PATH}: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"version", "exception_budget", "exceptions"}:
        raise PolicyError("policy must contain exactly version, exception_budget, and exceptions")
    if raw["version"] != 1:
        raise PolicyError("unsupported policy version")
    budget = raw["exception_budget"]
    entries = raw["exceptions"]
    if type(budget) is not int or not 0 <= budget <= MAX_EXCEPTION_BUDGET:
        raise PolicyError(f"exception_budget must be between 0 and {MAX_EXCEPTION_BUDGET}")
    if not isinstance(entries, list) or len(entries) != budget:
        raise PolicyError("exception count must equal exception_budget")

    rules: dict[str, ExceptionRule] = {}
    previous = ""
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "max_lines", "rationale"}:
            raise PolicyError(f"exceptions[{index}] has an invalid shape")
        path = entry["path"]
        parsed = PurePosixPath(path) if isinstance(path, str) else None
        if (
            parsed is None
            or parsed.is_absolute()
            or parsed.as_posix() != path
            or not path.startswith("src/")
            or parsed.suffix != ".rs"
            or parsed.name == "tests.rs"
            or "tests" in parsed.parts
        ):
            raise PolicyError(f"exceptions[{index}].path must be a production src/**/*.rs file")
        if path <= previous:
            raise PolicyError("exceptions must be sorted by path")
        previous = path
        max_lines = entry["max_lines"]
        if type(max_lines) is not int or not TARGET_LINES < max_lines <= LEGACY_CEILING_LIMIT:
            raise PolicyError(
                f"exceptions[{index}].max_lines must be between {TARGET_LINES + 1} "
                f"and {LEGACY_CEILING_LIMIT}"
            )
        rationale = entry["rationale"]
        if not isinstance(rationale, str) or len(rationale.strip()) < 30:
            raise PolicyError(f"exceptions[{index}].rationale must explain the boundary")
        if max_lines > NEW_MODULE_LIMIT and "legacy" not in rationale.lower():
            raise PolicyError(f"exceptions[{index}] above {NEW_MODULE_LIMIT} lines must be legacy")
        if path in rules:
            raise PolicyError(f"duplicate exception path: {path}")
        rules[path] = ExceptionRule(max_lines=max_lines, rationale=rationale.strip())
    return rules


def evaluate(modules: dict[str, int], rules: dict[str, ExceptionRule]) -> list[str]:
    violations: list[str] = []
    for path in rules.keys() - modules.keys():
        violations.append(f"{path}: exception is stale because the module does not exist")
    for path, lines in modules.items():
        rule = rules.get(path)
        if rule is None and lines > NEW_MODULE_LIMIT:
            violations.append(f"{path}: {lines} lines exceeds the {NEW_MODULE_LIMIT}-line limit")
        elif rule is None and lines > TARGET_LINES:
            violations.append(f"{path}: {lines} lines exceeds target without a reviewed exception")
        elif rule is not None and lines > rule.max_lines:
            violations.append(f"{path}: grew to {lines} lines above ceiling {rule.max_lines}")
        elif rule is not None and lines <= TARGET_LINES:
            violations.append(f"{path}: now fits the target; remove its stale exception")
        elif rule is not None and lines < rule.max_lines:
            violations.append(f"{path}: shrank to {lines} lines; lower ceiling {rule.max_lines}")
    return sorted(violations)


def main() -> int:
    try:
        modules = production_modules()
        rules = load_policy()
    except PolicyError as error:
        print(f"Invalid module-size policy: {error}", file=sys.stderr)
        return 2

    largest = sorted(modules.items(), key=lambda item: (-item[1], item[0]))
    print(f"Largest Rust production modules ({len(modules)} scanned):")
    for path, lines in largest[:REPORT_COUNT]:
        ceiling = f" [reviewed ceiling: {rules[path].max_lines}]" if path in rules else ""
        print(f"{lines:4}  {path}{ceiling}")
    print(
        "LOC is a review trigger, not a design score; split responsibilities before "
        "adding or widening an exception."
    )

    violations = evaluate(modules, rules)
    if violations:
        print("Module-size policy violations:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print(
        f"Module-size policy passed: target <= {TARGET_LINES}, new-module limit <= "
        f"{NEW_MODULE_LIMIT}, reviewed exceptions {len(rules)}/{MAX_EXCEPTION_BUDGET}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
