import json
import os
import subprocess
import sys


def test_cli_help_lists_save_resume_and_live_base_storage():
    result=subprocess.run([sys.executable,'-m','ats','--help'],capture_output=True,check=True)
    assert all(option.encode() in result.stdout for option in ('--save-state','--load-state','--base-batteries','--excavators'))


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
    c.base.equipment.panels[0].quantity=30
    c.fleet.models[0].equipment.bucket.volume_m3=.075
    path=tmp_path/'equipment.json';path.write_text(json.dumps(equipment_data(c)),encoding='utf-8')
    result=subprocess.run([sys.executable,'-m','ats','--headless','--mode','legacy','--equipment',str(path),'--steps','500'],capture_output=True,check=True)
    r=json.loads(result.stdout)
    assert r['base_mass_kg']==1040 and r['panel_area_m2']==30
    assert r['drones'][0]['payload_kg']==120
    assert r['drones'][0]['transported_volume_m3']>0
    assert abs(r['energy_error_j'])<1e-4 and abs(r['mass_error_kg'])<1e-7


def test_cli_quantity_overrides_loaded_recipe_and_saves_ten_physical_krots(tmp_path):
    from tests.test_excavator_fleet import config
    from ats.experiments import save_configuration
    from ats.checkpoint import load_state
    recipe=tmp_path/'recipe.json';save_configuration(recipe,config(3))
    state=tmp_path/'ten.atsstate'
    result=subprocess.run([sys.executable,'-m','ats','--headless','--configuration',str(recipe),
        '--excavators','10','--containers','80','--steps','55','--save-state',str(state)],capture_output=True,check=True)
    report=json.loads(result.stdout);s=load_state(state).simulation
    assert len(s.logistics.excavators)==10 and len(s.logistics.containers)==80
    assert len(report['excavators'])==10 and all(e['excavated_kg']>0 for e in report['excavators'])
    metric=report['excavation_cost']
    assert metric['currency']=='USD' and metric['capital_per_m3']>0
    assert json.loads(json.dumps(s.report(),allow_nan=False))==report


def test_cli_can_remove_krots_from_a_draft_before_initial_delivery(tmp_path):
    from tests.test_excavator_fleet import config
    from ats.experiments import save_configuration
    recipe=tmp_path/'recipe.json';save_configuration(recipe,config(3))
    result=subprocess.run([sys.executable,'-m','ats','--headless','--configuration',str(recipe),
        '--excavators','0','--steps','5'],capture_output=True,check=True)
    report=json.loads(result.stdout)
    assert 'excavator' not in report and report['energy_excavator_kwh']==0
    assert all(row['key']!='excavator' for row in report['equipment_cost_estimate']['rows'])
