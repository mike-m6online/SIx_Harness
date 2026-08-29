"""Parameterized generic hook scripts for the portable Claude Code harness.

Each script in this directory is a standalone hook executable: the harness
init tool bakes a concrete command line (interpreter + script path + the
required ``--project-root`` / ``--memory-dir`` flags with absolute paths)
into the target project's hook configuration. Nothing here discovers paths
at runtime and nothing here carries a project-specific default.

Ported from the origin project's harness scripts; the
originals remain untouched and in service there. Behavior is identical to
the originals modulo parameterization -- see each module's docstring for
the exact deltas. Scripts assessed but NOT ported are recorded in
``EXCLUDED.md``.
"""
