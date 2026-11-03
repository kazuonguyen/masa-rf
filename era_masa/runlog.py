"""Durable logs: console transcript, structured events, configuration, environment."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path

BANGKOK = timezone(timedelta(hours=7))


def timestamp():
    return datetime.now(BANGKOK).isoformat(timespec="milliseconds")


def run_directory(root="outputs", name="run"):
    stamp = datetime.now(BANGKOK).strftime("%Y%m%d-%H%M%S-%f")
    return Path(root) / f"{stamp}-{name}"


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


class RunLog:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self.stack = []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.console_file = (self.directory / "console.log").open("w", encoding="utf-8", buffering=1)
        self.event_file = (self.directory / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self.save("config.json", self.config)
        packages = {}
        for name in ("numpy", "scipy", "torch", "mmcv", "mmdet", "mmengine", "era-masa-oa"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self.save("environment.json", {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages, "timezone": "Asia/Bangkok"})
        for manager in (redirect_stdout(Tee(sys.stdout, self.console_file)),
                        redirect_stderr(Tee(sys.stderr, self.console_file))):
            manager.__enter__()
            self.stack.append(manager)
        self.event("run_started", directory=str(self.directory.resolve()))
        return self

    def event(self, kind, **values):
        event = {"time": timestamp(), "event": kind, **values}
        self.event_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.event_file.flush()

    def message(self, message, **values):
        print(f"[{timestamp()}] {message}", flush=True)
        self.event("message", message=message, **values)

    def save(self, name, data):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            traceback.print_exception(exc_type, exc, tb)
            self.event("run_failed", error_type=exc_type.__name__, error=str(exc))
            self.save("status.json", {"status": "failed", "time": timestamp(), "error": str(exc)})
        else:
            self.event("run_completed")
            self.save("status.json", {"status": "completed", "time": timestamp()})
        for manager in reversed(self.stack):
            manager.__exit__(exc_type, exc, tb)
        self.console_file.close()
        self.event_file.close()
        return False
