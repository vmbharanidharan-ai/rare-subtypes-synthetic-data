"""Backward-compatible entrypoint for preprocess stage."""

from rare_synth.cli import main


if __name__ == "__main__":
    import sys

    sys.argv = ["rare_synth", "preprocess", "--config", "configs/default_uvm.yaml", "--root", "."]
    main()
