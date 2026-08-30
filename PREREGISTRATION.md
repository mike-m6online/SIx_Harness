# Pre-registered ablation protocol — v1, 2026-08-30

This protocol is registered by dated, tagged commit (`prereg-ablation-v1`)
**before any ablation run**, and is referenced by the paper *Memory as
Infrastructure*. Any later change will be recorded here as a dated
amendment, never by rewriting this text.

Both ablations run on a live deployment of this kit, with the
`wrapper_invocations` / `hook_heartbeat` / `embed_degradation` stores as
the measurement instruments. Workload is reported for every window
(prompts, intent-bearing prompts, hook invocations), and all comparisons
are made as **rates per intent-bearing prompt**, never raw counts —
observed workload varies several-fold between periods.

## A1 — injection off (14 days)

**Intervention:** the prompt-injection layer's *emission* is disabled.
Intent detection, relevance gating, and all telemetry remain on, so every
suppressed injection is recorded as a **would-have-fired** row and the
denominator is intact.

**Primary metric — re-proposal incidents**, mechanically defined: a
prompt with build intent for which (i) a would-have-fired DO-NOT-REBUILD
record names an existing subsystem, and (ii) the session subsequently
begins a new implementation overlapping that subsystem's documented
scope. Condition (ii) is scored independently by two raters (the human
operator and the agent), each blind to the other's rating; disagreements
are reported as disagreements, not resolved away. A would-have-fired
record alone is **delivery**, not prevention, and is never counted as an
incident by itself.

**Secondary metrics:** time from first mention of a capability to
discovery that it exists; stale-claim incidents that reach a commit
before being corrected.

**Comparison:** the 14 days immediately preceding the ablation, same
metrics, same raters.

**Decision rule:** the injection layer's value on this axis is
*supported* if the ablated window shows ≥2 mechanically-scored
re-proposal incidents and the intact window shows fewer at comparable
intent-bearing volume; *refuted on this axis* if the ablated window shows
0 incidents while ≥5 would-have-fired DO-NOT-REBUILD records accrue;
otherwise *inconclusive*, reported as such. Latency and interruption
counts are reported but are **not** creditable to the ablated arm:
disabling emission reduces both by construction.

## A2 — gate off (14 days)

**Intervention:** the memory-health gate is not run at session starts.
Heartbeat and degradation telemetry still record, so degradations remain
visible in retrospect.

**Primary metric — detection latency:** for each naturally occurring
degradation (classes already observed in production: cold embedding
server, stale ingest watermark, aged structural-query corpus), the time
from its first evidence row to the first remediation action.

**Decision rule:** the gate's value is *supported* if ≥2 degradations
occur in the window and their median detection latency exceeds the
intact configuration's boundary-catch median by ≥4×; *inconclusive* if
fewer than 2 degradations occur (the record suggests a base rate of
roughly one per multi-day idle gap, so a quiet fortnight is possible and
proves nothing either way).

## Reporting commitment

Results will be reported regardless of direction, with full metric
tables and both raters' scores. Anyone may run this protocol on their
own project with this kit; we will link independent runs from the README.
