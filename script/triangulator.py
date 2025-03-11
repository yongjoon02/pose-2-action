import os
import sys

# Add src folder to Python module search path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

from triangulation import main

if __name__ == "__main__":
    main()
