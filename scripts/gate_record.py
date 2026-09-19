#!/usr/bin/env python3
"""Turn a measured gate log (or cells file) into a certifiable gate record.

Certification itself stays in `go1_hier_sim.checkpoints.certify_checkpoint`; this
wrapper only produces the measured record.
"""

from go1_hier_sim.gate import main

if __name__ == "__main__":
    raise SystemExit(main())
