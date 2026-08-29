# Excluded hooks

Source scripts from the origin project's harness that were assessed for the kit but NOT ported, one line each with the reason. Everything else under `hooks/` was ported with parameterization deltas documented in each module's docstring.

- `context_monitor.py` — SKIPPED (meaningfully origin-coupled): its entire useful output is operator guidance hardwired to the retired agent-army workflow (checkpoint MCP calls — deprecated even on the origin project), its transcript discovery is pinned to the origin project's `~/.claude/projects/<slug>` dir, its singleton `%TEMP%/claude_context_monitor.json` state file collides across concurrently running projects, and its token constants (200K limit / 50K baseline / 2.0 bytes-per-token) are calibrations from the origin project's sessions — a correct generic version is a redesign of the messages and state model, not a parameterization.
- `deploy_check_hook` — excluded by design: project-specific deploy gating; each project supplies its own.
- `deploy_log_hook` — excluded by design: project-specific deploy logging; each project supplies its own.
- `cuda_review_hook` — excluded by design: project-specific CUDA-engine review gate; each project supplies its own.
- `gen_project_state` — excluded by design: derives per-module state from the project's own code layout; each project supplies its own (the kit ships a template stub via the harness init tool's templates — sibling task).
