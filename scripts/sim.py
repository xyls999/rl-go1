"""Workspace command entry point; simulator startup stays inside the CLI."""

import faulthandler
import signal

from go1_hier_sim.cli import main

if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)

if __name__ == "__main__":
    main()
