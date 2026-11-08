import subprocess
import sys


def test_can_run_as_python_module():
    """Run the CLI as a Python module."""
    result = subprocess.run(
        [sys.executable, "-m", "jordan_form", "--help"],
        check=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert b"jordan-form [OPTIONS]" in result.stdout
