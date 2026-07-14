from __future__ import annotations

import sys
import unittest


def main() -> None:
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("Automated tests do not validate acoustic accuracy.")
    print("New in-car measurements are required.")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
