"""Backward-compatible command location for the README entry point."""

from tdpa.tools.verify_physics import main, verify

__all__ = ["main", "verify"]

if __name__ == "__main__":
    main()

