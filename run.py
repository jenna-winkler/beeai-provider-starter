import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from beeai_agents.server import main

if __name__ == "__main__":
    main()