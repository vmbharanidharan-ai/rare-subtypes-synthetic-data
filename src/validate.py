"""Backward-compatible entrypoint for validation stage."""

from rare_synth.cli import main


if __name__ == "__main__":
    import sys

    sys.argv = ["rare_synth", "validate", "--config", "configs/default_uvm.yaml", "--root", "."]
    main()
