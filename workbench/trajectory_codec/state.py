import json
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


class State:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "state.json"
        self.value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def update(self, **values):
        self.value.update(values, updatedAt=datetime.now(timezone.utc).isoformat())
        atomic_json(self.path, self.value)

    def stopped(self):
        return (self.output / "stop.request").exists()


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)
