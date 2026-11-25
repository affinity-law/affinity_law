"""Launch the local research workbench from any checkout location."""
import sys

sys.dont_write_bytecode = True

from server import main

if __name__ == '__main__':
    main(open_browser=True)
