import os
import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported in Vercel serverless environment
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from chargeshield.api.main import app
