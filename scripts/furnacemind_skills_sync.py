"""Bring the FurnaceMind skills table in line with the code's quick skills (user decision 13 Nov 2026).

Since 13 Nov 2026 the code's seven plant skills are offered whether or not the skills table holds rows for them, and
the rows of the retired Unit Cost, Shift to Best and Heatloads are ignored (``agents.furnacemind.skill_registry``).
This script tidies the table so page 7's skill sidebar lists what the chat offers:

* the rows of the retired skills are set inactive (kept, so they can be restored);
* a row is added for every code skill without one (active, source ``built_in``, metadata slug / order / symbol /
  handler / display_template), so an administrator can rename or hide it there;
* uploaded skills are left alone.

Dry run by default; ``--apply`` writes. Run it after the release with the plant skills is deployed everywhere: the
table is shared by every running copy of the app (``DATABASE_URL``), and an older copy does not know the new handlers.

    python scripts/furnacemind_skills_sync.py            # the plan
    python scripts/furnacemind_skills_sync.py --apply    # write it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changes (default: print the plan)")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    from agents.furnacemind.skill_registry import (
        _BUILTIN_SKILLS,
        _RETIRED_HANDLERS,
        _RETIRED_SLUGS,
        _definition_from_row,
        _metadata_for,
        _string_value,
    )
    from services.furnacemind import _skill_repository

    repo = _skill_repository().get("skill_repository")
    if repo is None:
        print("The skills table is not reachable (DATABASE_URL): nothing to do.")
        return 1
    rows = list(repo.list_skills(active_only=False))
    plan: list[tuple[str, str, dict]] = []
    present: set[str] = set()
    for index, row in enumerate(rows):
        meta = _metadata_for(row)
        definition = _definition_from_row(row, fallback_order=1000 + index)
        present.add(definition.slug)
        handler = _string_value(meta.get("handler") or meta.get("handler_name"))
        retired = definition.slug in _RETIRED_SLUGS or handler in _RETIRED_HANDLERS
        if retired and getattr(row, "is_active", True):
            plan.append(("deactivate", f"{row.name} ({row.skill_id})", {"skill_id": row.skill_id}))
    for skill in _BUILTIN_SKILLS:
        if skill.slug in present:
            continue
        plan.append(
            (
                "add",
                f"{skill.name} ({skill.slug})",
                {
                    "name": skill.name,
                    "symbol": skill.symbol,
                    "description": skill.description,
                    "instruction": skill.instruction,
                    "source_type": "built_in",
                    "is_active": True,
                    "metadata": {
                        "slug": skill.slug,
                        "order": skill.order,
                        "symbol": skill.symbol,
                        "handler": skill.handler,
                        "display_template": skill.display_template,
                    },
                },
            )
        )
    if not plan:
        print("The skills table already matches the code's skills.")
        return 0
    for action, label, _ in plan:
        print(f"{'WRITE' if args.apply else 'plan '} {action:10} {label}")
    if not args.apply:
        print("Dry run: add --apply to write (after the release with the plant skills is deployed).")
        return 0
    for action, _label, payload in plan:
        if action == "deactivate":
            repo.update_skill(skill_id=payload["skill_id"], is_active=False)
        else:
            repo.create_skill(**payload)
    print(f"{len(plan)} change(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
