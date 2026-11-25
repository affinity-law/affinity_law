"""Set up and launch the local forecasting workbench with Python 3.11."""

import os
from pathlib import Path
import subprocess
import sys
import venv
from importlib import metadata


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
ENVIRONMENT = ROOT / "4html" / ".runtime" / "venv"


def ready():
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        pin = line.partition("#")[0].strip()
        if not pin:
            continue
        name, version = pin.split("==")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            return False
        if installed.partition("+")[0] != version:
            return False
    return True


def main():
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise SystemExit("Please run this launcher with 64-bit Python 3.11.")
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        return subprocess.call([sys.executable, "-B", str(ROOT / "4html/start.py"), "--help"])
    if not ready():
        if Path(sys.prefix).resolve() != ENVIRONMENT:
            exe = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not exe.is_file():
                print("Creating the workbench's private Python environment...", flush=True)
                venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)
            return subprocess.call([str(exe), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])
        print("First-time setup: downloading PyTorch and the project dependencies.", flush=True)
        pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install"]
        if sys.platform != "darwin":
            torch_pin = next(pin.strip() for pin in REQUIREMENTS.read_text().splitlines()
                            if pin.startswith("torch=="))
            subprocess.run([*pip, torch_pin, "--index-url", "https://download.pytorch.org/whl/cpu"],
                           check=True)
        subprocess.run([*pip, "-r", str(REQUIREMENTS)], check=True)
        if not ready():
            raise SystemExit("Installed versions do not match requirements.txt. Check the pip output above.")
    return subprocess.call([sys.executable, "-B", str(ROOT / "4html/start.py"), *sys.argv[1:]],
                           cwd=ROOT)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Workbench setup failed: {exc}")
