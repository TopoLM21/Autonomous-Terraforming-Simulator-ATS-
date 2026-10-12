import json
import os
import subprocess
import sys


def test_headless_json_in_legacy_console_and_utf8_file(tmp_path):
    environment = dict(os.environ, PYTHONIOENCODING="cp1252")
    path = tmp_path/"report.json"
    result = subprocess.run(
        [sys.executable, "-m", "ats", "--headless", "--steps", "0", "--json", str(path)],
        env=environment, capture_output=True, check=True,
    )
    report = json.loads(result.stdout.decode("cp1252"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert report == saved
    assert report["elapsed_display"] == "0 д 00:00:00"
    assert report["solar_clock"] == "09:00:00"
