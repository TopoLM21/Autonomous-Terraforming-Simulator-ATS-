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


def test_equipment_profile_cli_uses_inventory_and_reports_production(tmp_path):
    from ats.simulation import Simulation
    from ats.equipment import equipment_data
    s=Simulation();c=s.config
    c.base.equipment.panels[0].quantity=15
    c.fleet.models[0].equipment.bucket.volume_m3=.075
    path=tmp_path/'equipment.json';path.write_text(json.dumps(equipment_data(c)),encoding='utf-8')
    result=subprocess.run([sys.executable,'-m','ats','--headless','--equipment',str(path),'--steps','500'],capture_output=True,check=True)
    r=json.loads(result.stdout)
    assert r['base_mass_kg']==1040 and r['panel_area_m2']==30
    assert r['drones'][0]['payload_kg']==120
    assert r['drones'][0]['transported_volume_m3']>0
    assert abs(r['energy_error_j'])<1e-4 and abs(r['mass_error_kg'])<1e-7
