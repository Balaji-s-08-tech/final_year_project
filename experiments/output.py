"""Tee standalone CLI output while keeping subprocess streams separate."""
import contextlib
import logging
import sys


class Tee:
    def __init__(self, original, retained):
        self.original, self.retained = original, retained

    def write(self, text):
        self.retained.write(text)
        self.retained.flush()
        return self.original.write(text)

    def flush(self):
        self.retained.flush()
        self.original.flush()


@contextlib.contextmanager
def record_output(run):
    with (run.path / "stdout.log").open("a") as stdout, (run.path / "stderr.log").open("a") as stderr:
        handler = logging.StreamHandler(stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, stdout)), contextlib.redirect_stderr(Tee(sys.stderr, stderr)):
                yield
        finally:
            logging.getLogger().removeHandler(handler)
