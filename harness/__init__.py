"""SIx Harness kit -- portable Claude Code harness bootstrap.

This package carries the ``harness`` console command. Its one subcommand,
``harness init``, wires a project for the kit's Claude Code harness:

* creates the Claude Code per-project memory skeleton (``MEMORY.md`` under
  ``~/.claude/projects/<slug>/memory/``),
* creates the append-only arc ledger (``.superpowers/sdd/progress.md``),
* seeds a ``CLAUDE.md`` constitution from the kit template (never
  overwriting an existing one),
* MERGES the claude-mem + kit-hook entries into the project's Claude Code
  settings JSON without clobbering anything that is already there.

See :mod:`harness.init` for the full behavior contract and
``README.md`` at the kit root for the operator-facing story.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
