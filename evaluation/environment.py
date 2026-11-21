"""Prints the software and hardware environment used for an evaluation run."""
import os
import platform
import shutil
import subprocess
from importlib import metadata

print("python", platform.python_version())
print("platform", platform.platform())
print("machine", platform.machine(), "cpus", os.cpu_count())
for pkg in ("numpy", "scikit-learn", "pillow", "pyyaml", "pydantic", "fastapi", "uvicorn", "httpx", "pytest"):
    try:
        print(pkg, metadata.version(pkg))
    except metadata.PackageNotFoundError:
        print(pkg, "not installed")
for tool in ("ffmpeg", "tesseract"):
    path = shutil.which(tool)
    ver = subprocess.run([tool, "-version" if tool == "ffmpeg" else "--version"], capture_output=True, text=True).stdout.splitlines()[:1] if path else ["missing"]
    print(tool, ver[0] if ver else "")
print("seeds", "pipeline is deterministic; bootstrap seed=0; red-team seed=0")
