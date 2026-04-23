"""Backward-compatible entrypoint for training stage."""

from rare_synth.cli import main


if __name__ == "__main__":
    import sys

    sys.argv = ["rare_synth", "train", "--config", "configs/default_uvm.yaml", "--root", "."]
    main()
