from typer.testing import CliRunner

from jordan_form.cli import app

runner = CliRunner()


def test_help():
    """The help message includes the CLI name."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_main() -> None:
    """The CLI prints the computed Jordan chains."""
    result = runner.invoke(app, ["{{x - 1}}", "1"])
    assert result.exit_code == 0
    assert "Algebraic multiplicity:" in result.stdout
    assert "Chain " in result.stdout
