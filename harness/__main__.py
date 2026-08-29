"""``python -m harness`` -- module entry for the harness kit CLI.

Delegates to :func:`harness.init.main`, the same function the
``harness`` console script targets, so both invocation forms behave
identically.
"""

from harness.init import main

if __name__ == "__main__":
    raise SystemExit(main())
