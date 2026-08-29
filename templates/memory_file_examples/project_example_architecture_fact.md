---
name: project-example-architecture-fact
description: "Load-bearing architectural fact -- ONE SENTENCE a search hit can act on alone, e.g.: 'The dialogue model is the brain's PRIVATE inner monologue for self-conditioning, NOT inter-agent communication; five consumers, only one provides outcome-conditioned feedback.' Read when <the trigger question this file answers>."
metadata:
  node_type: memory
  type: project
---

# <Fact title>

<!--
PROJECT files carry load-bearing architectural facts, decisions, and
plans -- the "what is true about THIS system's design" tier. Conventions:

* Filename starts with `project_`; MEMORY.md links it as ONE line under
  "Load-bearing architectural facts" or "START HERE".
* The `description` should name the TRIGGER: which recurring question
  this file answers ("read when the user asks about X"), because that is
  how retrieval decides to surface it.
* State the fact first, evidence second. Cite file:line and commit
  hashes for every code claim -- and remember the ground-truth hierarchy:
  a reader must verify named symbols against current code before ACTING
  on them, so give them greppable names.
* Record what the fact RULES OUT as explicitly as what it establishes
  ("NOT inter-agent communication") -- negative space prevents
  re-proposal of rejected designs.
-->

**The fact:** <the architectural fact in one or two sentences, including
what it explicitly rules out>.

**Evidence / provenance:**
- <file:line or module reference for each code-level claim>
- <commit hash / dated doc for the decision that established it>
- <the ruling: who approved it and when, if it was an operator decision>

**Consequences for future work:**
- <what designs this fact forbids or requires>
- <what to check before building against it (the greppable symbols)>
