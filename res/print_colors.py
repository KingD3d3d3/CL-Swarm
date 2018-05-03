import sys

PRINT_RED     = "\033[1;31m"
PRINT_BLUE    = "\033[1;34m"
PRINT_CYAN    = "\033[1;36m"
PRINT_GREEN   = "\033[0;32m"
PRINT_RESET   = "\033[0;0m"
PRINT_BOLD    = "\033[;1m"
PRINT_REVERSE = "\033[;7m"

def printColor(color=PRINT_BLUE, message=""):
    sys.stdout.write(color)
    print(message)
    sys.stdout.write(PRINT_RESET)