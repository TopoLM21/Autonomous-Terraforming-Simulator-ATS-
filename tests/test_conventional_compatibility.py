"""Read a real pre-dimensions/pre-ramp-rework version-22 checkpoint.

loaded.atsstate was generated, without deleting or editing saved fields, by
bb66a6cadfffe0375d465b24afacedaf4d4f5f3b in a separate archived checkout:

    from tests.test_mixed_setup import running_site
    from ats.checkpoint import save_state
    simulation, fleet = running_site(soil=True)
    save_state('loaded.atsstate', simulation)

The original machines were working for 750 s, with actual layered excavation
and 472.16416288857863 kg in a truck. No new placement is requested on load.
"""
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from ats.checkpoint import load_state,save_state


FIXTURE=Path(__file__).parent/'fixtures'/'conventional-v22'/'loaded.atsstate'


def _array(archive,nodes,reference):
    with archive.open(nodes[reference['ref']]['file']) as payload:
        return np.load(payload,allow_pickle=False)


def _balanced(simulation):
    fleet=simulation.conventional;terrain=simulation.terrain
    fleet.validate()
    assert abs(terrain.mass_change+fleet.carried_mass_kg)<1e-7
    np.testing.assert_allclose(terrain.soil.volume_delta_by_material*terrain.soil.profile.densities+
                               fleet.carried_soil.masses,0.,atol=1e-7)
    assert abs(fleet.report()['fuel_balance_error_l'])<1e-7


def test_authentic_v22_manual_loaded_fleet_keeps_its_geometry_clock_and_materials():
    with ZipFile(FIXTURE) as archive:
        manifest=json.loads(archive.read('manifest.json'));nodes=manifest['nodes']
        assert manifest['version']==22
        saved={node['type']:node['attrs'] for node in nodes if node.get('kind')=='object'}
        cfg=saved['ats.conventional.ConventionalConfig'];old=saved['ats.conventional.ConventionalFleet']
        assert not {'auto_work_area','excavator_dimensions_m','truck_dimensions_m'}&cfg.keys()
        machine_nodes=[node['attrs'] for node in nodes if node.get('type')=='ats.conventional.GroundMachine']
        assert all(not {'body_dimensions_m','heading_rad'}&node.keys() for node in machine_nodes)
        simulation=load_state(FIXTURE).simulation;fleet=simulation.conventional;terrain=simulation.terrain
        assert simulation.time==750. and fleet.sim is simulation and fleet.terrain is terrain
        assert fleet.config.auto_work_area is False
        assert fleet.config.excavator_dimensions_m==(6.,3.,3.)
        assert fleet.config.truck_dimensions_m==(7.,2.8,3.2)
        assert fleet.carried_mass_kg==472.16416288857863 and fleet.fuel_stock_l==old['fuel_stock_l']
        assert [m.id for m in fleet.machines]==[node['id'] for node in machine_nodes]
        for machine,node in zip(fleet.machines,machine_nodes):
            np.testing.assert_array_equal(machine.position,_array(archive,nodes,node['position']))
            np.testing.assert_array_equal(machine.home,_array(archive,nodes,node['home']))
            assert machine.fuel_l==node['fuel_l'] and machine.cargo_kg==node['cargo_kg']
            assert tuple(machine.dimensions_m)==((6.,3.,3.) if machine.kind=='excavator' else (7.,2.8,3.2))
        for key in ('height','reference','target'):
            np.testing.assert_array_equal(getattr(terrain,key),_array(archive,nodes,saved['ats.terrain.Terrain'][key]))
        np.testing.assert_array_equal(fleet.ramp_surface,_array(archive,nodes,old['ramp_surface']))
        np.testing.assert_array_equal(fleet.ramp_mask,_array(archive,nodes,old['ramp_mask']))
        assert fleet.config.work_rect==(35.,40.,60.,20.)
        np.testing.assert_array_equal(fleet.base_position,[12.,12.,0.])
        _balanced(simulation)


def test_authentic_v22_fleet_continues_and_new_snapshot_resumes_exactly(tmp_path):
    simulation=load_state(FIXTURE).simulation
    path=tmp_path/'continued.atsstate';save_state(path,simulation)
    restored=load_state(path).simulation
    assert simulation.report()==restored.report()
    for _ in range(30):simulation.step();restored.step()
    assert simulation.time==restored.time==780.
    assert simulation.report()==restored.report()
    np.testing.assert_array_equal(simulation.terrain.height,restored.terrain.height)
    assert simulation.conventional.config.work_rect==(35.,40.,60.,20.)
    assert [m.id for m in simulation.conventional.machines]==['excavator_1','truck_1','truck_2']
    _balanced(simulation);_balanced(restored)
