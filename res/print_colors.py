
import sys

PRINT_RED = "\033[1;31m"
PRINT_BLUE = "\033[1;34m"
PRINT_CYAN = "\033[1;36m"
PRINT_GREEN = "\033[0;32m"
PRINT_RESET = "\033[0;0m"
PRINT_BOLD = "\033[;1m"
PRINT_REVERSE = "\033[;7m"

def print_color(color=PRINT_BLUE, msg=""):
    """
        Print a message in the console with the specified color
        :param color: color to print
        :param msg: text to print
    """
    sys.stdout.write(color)
    print(msg)
    sys.stdout.write(PRINT_RESET)
