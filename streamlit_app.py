import os
import sys
from pathlib import Path

# Ensure root directory is on python path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load and execute the dashboard app
app_file = ROOT_DIR / "dashboard" / "app.py"
with open(app_file, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_file), "exec")
    exec(code, globals())
