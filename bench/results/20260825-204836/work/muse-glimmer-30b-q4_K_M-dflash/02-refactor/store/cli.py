import sys

from store.source import fetch_data
from store.report import names


def main(argv):
    if "--names" in argv:
        print(",".join(names()))
    else:
        print(len(fetch_data()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
