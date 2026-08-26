import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cli.main import main


def main_entry():
    sys.exit(main())

if __name__ == "__main__":
    sys.exit(main())
