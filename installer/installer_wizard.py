import os
import sys
import subprocess
import shutil
import tempfile

REPO_URL = "https://github.com/bneiG1/BCI-Unicorn-Hybrid-Black_P300_Speller"
EXE_OUTPUT_DIR = os.path.abspath(os.getcwd())


def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(1)


def main():
    print("--- BCI P300 Speller Installer Wizard ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning repository into {tmpdir} ...")
        run_command(["git", "clone", REPO_URL, tmpdir])
        build_script = os.path.join(tmpdir, "build_installer.py")
        if not os.path.exists(build_script):
            print("build_installer.py not found in the repository!")
            sys.exit(1)
        print("Running build_installer.py ...")
        run_command([sys.executable, build_script], cwd=tmpdir)
        dist_dir = os.path.join(tmpdir, "dist_installer")
        if not os.path.exists(dist_dir):
            print("dist_installer directory not found after build!")
            sys.exit(1)
        exe_files = [f for f in os.listdir(dist_dir) if f.endswith(".exe")]
        if not exe_files:
            print("No .exe files found in dist_installer!")
            sys.exit(1)
        for exe in exe_files:
            src = os.path.join(dist_dir, exe)
            dst = os.path.join(EXE_OUTPUT_DIR, exe)
            shutil.copy2(src, dst)
            print(f"Copied {exe} to {EXE_OUTPUT_DIR}")
    print("\nAll done! The .exe file(s) are in your current directory.")

if __name__ == "__main__":
    main()
