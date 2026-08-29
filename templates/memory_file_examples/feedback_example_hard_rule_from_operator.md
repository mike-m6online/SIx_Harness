---
name: feedback-example-hard-rule-from-operator
description: "Hard rule from <operator> <YYYY-MM-DD>: ONE-SENTENCE statement of the rule and when it applies, e.g.: 'Never launch long jobs with nohup over SSH -- use screen -dmS; session teardown kills nohup children silently.'"
metadata:
  node_type: memory
  type: feedback
---

# <Rule title>

<!--
FEEDBACK files record hard rules and corrections from the operator --
things a human said after catching a failure mode. Conventions:

* Filename starts with `feedback_`; MEMORY.md links it as ONE line under
  the "Hard rules" section.
* Open with WHO said it, WHEN, and after WHAT incident -- provenance is
  what separates a hard rule from an agent's own preference.
* Quote the operator verbatim where possible.
* Then explain WHY (the mechanism of the failure the rule prevents) and
  HOW TO APPLY (the concrete behavioral test an agent can run on itself).
* A feedback file is point-in-time: if the rule is later superseded, add
  a dated superseded-by note at the top rather than deleting the file.
-->

<Operator>, <YYYY-MM-DD>, after <the incident that earned the rule>:

> "<verbatim quote of the operator's correction>"

**Why:** <the mechanism -- what actually goes wrong when the rule is
broken, and why it is invisible in the moment>.

**How to apply:**

1. <the concrete behavioral test, e.g. "before proposing X, check Y">
2. <the escalation path when the rule blocks progress: surface to the
   operator, never decide unilaterally to break it>

Source: <operator>, <YYYY-MM-DD>, <session/doc/commit pointer>.
