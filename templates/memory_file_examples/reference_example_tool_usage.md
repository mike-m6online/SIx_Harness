---
name: reference-example-tool-usage
description: "How to use <tool>, the project's <what it does>. USE IT for <the class of question/task it serves> instead of <the failure mode it replaces, e.g. re-deriving by hand>."
metadata:
  node_type: memory
  type: reference
---

# <Tool name>: what it is and how to invoke it

<!--
REFERENCE files document tools and external resources so sessions USE
them instead of concluding they do not exist or re-implementing them.
Conventions:

* Filename starts with `reference_`; MEMORY.md links it as ONE line
  under "Tools & references (USE THESE)".
* Lead with the existence claim and the anti-dismissal note: agents'
  most common failure with project tooling is deciding it "isn't
  installed" -- say where it lives and how to prove it is alive.
* Give the EXACT invocation, including the platform quirks that make the
  naive invocation fail (encoding flags, PATH gaps, required env vars).
  Copy-pasteable beats descriptive.
* End with WHEN TO USE: the trigger questions that should route to this
  tool by default.
-->

`<tool>` is a real, installed tool: <what it is, where it lives (path /
package), and one line of proof it is alive (health command)>. Do NOT
conclude it "isn't installed."

**Invocation (<platform notes>):**
```
<the exact command lines, including encoding/env workarounds>
```

**Commands / options that matter:** <the short list actually used here,
one line each>.

**WHEN TO USE (default to using it):** <the trigger questions -- "any
'have we done X' question", "before proposing something that may already
exist" -- that should route here by default>.
