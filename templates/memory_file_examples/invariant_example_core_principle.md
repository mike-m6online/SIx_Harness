---
name: invariant-example-core-principle
description: "ONE-SENTENCE statement of the invariant strong enough to act on alone, e.g.: 'The ONLY learning signal is prediction error -- no rewards, no shaping.' Re-read at every arc boundary."
metadata:
  node_type: memory
  type: invariant
---

# <Invariant title> (<pointer to the principle's canonical source, e.g. PRINCIPLES.yaml key>)

<!--
INVARIANT files are the FIRST-CLASS tier of the memory dir:

* Filename MUST start with `invariant_` -- the MEMORY.md `## INVARIANTS`
  section links every one of them, and the session-start ritual is to
  actively RE-READ them at every arc boundary and before any design pass
  (passive injection gets skimmed).
* Keep the set small (2-6 files). An invariant is a principle the project
  cannot recover if it decays -- the load-bearing "why", not an
  operational tip (that is a feedback_/process_ file).
* The `description` frontmatter must carry the invariant's full one-line
  statement: search hits often surface only the description, so it has to
  stand alone.
* Body structure below: statement in bold, then the narrow valid
  exceptions, then the tempting violations NAMED explicitly -- naming the
  violation is what makes the invariant enforceable.
-->

**The invariant, stated in full:** <the rule, in one or two bold
sentences, with the exact quantities/formulas that make it checkable>.

Valid exceptions are narrow: <list them explicitly, or write "none">.
Anything else -- <name the tempting violations concretely, e.g. "rewards
for X, penalties for Y, a shortcut that hardcodes Z"> -- is a violation.

**Why this is load-bearing:** <the one-paragraph story of what breaks
without it; cite the incident that earned the rule if there is one>.

Re-read at every arc boundary.
