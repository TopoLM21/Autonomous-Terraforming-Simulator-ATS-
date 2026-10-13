# ATS 2.0: implementation and assumptions

The extension uses the existing `Simulation`, `Terrain`, `Drone`, `SolarBase`,
`Planner`, `Dust`, laser links and PySide6 GUI. `Config.logistics=None` retains
the original engine and JSON equipment/brush-plan formats.

| Existing class | Extension |
| --- | --- |
| `equipment.BatteryModule` (recipe with quantity) | `inventory.BatteryModule` (individual physical ID) |
| `Drone` | Mounted battery IDs, interchangeable tool and reserved transport job |
| `Terrain` | Single conservative `scoop` / `deposit` API used by all machines |
| `SolarBase` | Shared finite bus and physical charging ports |
| `Planner` | Original direct excavation planner, supervised by logistics dispatcher |
| `Simulation` | Optional finite logistics subsystem, original fixed one-second step |
| MainWindow | Additional inventory, excavator, energy and dispatcher views |

Baseline: 108 tests passed in 111.06 s before changes. First medium-scenario
flight with identical installed laser hardware: one battery 306 s off / 380 s
on; two batteries 491 s off / 600 s on. Full records: `ats2-baseline.json`.
Engineering defaults from the supplied specification are scenario assumptions,
not certified specifications of actual construction equipment.

## Resources and time

`inventory.py` stores physical instances, locations, holders and atomic resource
reservations. Equipment recipes remain compatible with version-1 equipment JSON.
All physical batteries are enumerated, including the base bank, drone mounts,
spares and the four excavator packs. Compatible drone packs share one warehouse
pool; their IDs do not change when another drone mounts them. Module count is
distinct from installed-slot count. All stored energy is included in the initial
world budget; initially full spares are explicit resources.

`logistics.py` supervises original drones using their finite speed, hover/winch
power and swept collision checks. It reserves each item and executor, ages queued
priorities, delivers empty containers, hauls filled ones, returns empties, and
delivers/retrieves individual batteries and the common rig. A departing drone keeps
its landing pad until it reaches cruise height, preventing opposite traffic in
the same vertical shaft. The selected laser never enters the guaranteed battery
budget: a mission must remain feasible if its beam is lost.

`excavator.py` uses the original `Terrain.scoop` with reach and local-depth masks.
Sand first moves into its finite bucket, then into one reserved physical container.
No hidden hopper exists. Tracks/supports, slope and an approximate restoring versus
digging moment constrain access. The machine does not excavate its own support
footprint. Worksite choice penalizes unexcavated soil under tracks and excludes
stationary pseudo-moves. Inaccessible fronts cause a tracked move or collective
air lift; absence of a viable site is explicit. While batteries are unavailable,
60 W control idle changes to modelled 1 W sleep below 10% SOC. The 0.1 kWh buffer
only powers control, never mining; recharging it from installed packs pays 10% loss.

`batteries.py` runs timed sequential side-dock swaps and base drone swaps. Universal
ports charge **removed** modules, one ID per port; a fifth waits. Mounted drone
banks use the preserved landing-dock circuit. Their BMS shares accepted dock input
between mounted packs. An unported removed pack never receives a share of another
pack's charge. Contact docks, universal ports, mechanics, electronics and optical
links share finite generation/storage and the inverter limit. Mechanical docks
can use current sunlight even without a base storage battery. A delivered pack can
power the side-dock service connector if the control buffer is empty.

For dry-body relocation, batteries are physically removed first. The rig must
arrive by drone. All participants are reserved; a quasi-static cable-angle/load,
20% thrust margin and conservative energy check precede flight. Formation members
move simultaneously; the swept lower cargo point and each aircraft are checked
against terrain. Physical cargo shares and cable-equivalent load are reported
separately. Removed partially used packs go back to charging; they cannot strand
on the rack as supposedly ready modules.

The soil ledger is terrain + excavator bucket + all containers + direct grabbers.
Container tare and rigging never enter soil mass. The electrical ledger includes
all stored pack energies, control buffer, useful consumption, contact/optical
losses and curtailed sunlight. Transporting a battery does not transfer its energy
to the carrying drone. Brush fill deliveries accept only available target capacity;
the remainder stays in the container and is delivered to the dump. Landing for a
plan change finishes the bucket/container chain before pausing.

## Model defaults

| Parameter | Value | Status |
| --- | ---: | --- |
| Excavator dry mass / installed packs | 160 kg / 2 × 11 kg | Model |
| Excavator packs in the world | 4 × 1.5 kWh | Explicit finite inventory |
| Bucket / fill / cycle | 15 L / 0.75 / 35 s | Model |
| Digging average / peak constraint | 2.5 / 6 kW | Model |
| Reach / local depth / tracked speed | 2 m / 1.2 m / 0.2 m/s | Model |
| Grade / support half-width / tipping margin | 10° / 0.6 m / 1.5 | Model |
| Ordinary idle / sleep | 60 W / 1 W | Model |
| Side-dock swap / drone swap / tool change | 60 / 60 / 45 s | Model |
| Removed-pack ports / input limit | 4 / 1.5 kW each | Model, shared base budget |
| Drone pack input / discharge rating | 4 / 60 kW | Compatibility assumptions for existing heavy flight model |
| Base bank charge / discharge rating | 2C / 10C | Model, also constrained by inverter |
| Base inverter / common charge input | 100 / 8 kW | Model / existing charge input preserved |
| Common rig / cable angle / lift margin | 12 kg / 30° / 1.2 | Model |
| Container tare / volume / gross | 5 kg / 55 L / 90 kg | Model |
| Large field | 200 m², mass from panel recipes, 300 s deployment at 200 W | Separate stationary scenario |

All settings are in `LogisticsConfig` or existing equipment/configuration classes.
Internal states use seconds, metres, kilograms, joules and watts; user-facing
capacity inputs marked kWh are converted into joules on instantiation.

## Laser profiles

Legacy electrical/optical parameters remain intact, including custom saved values.
Reference: 5 kW source input, 40% source efficiency, 75% reference capture, 40%
receiver efficiency, 1.5 kW optical cap, 0.8 kW electrical peak rating, 2.8 kg
receiver and one shared 300 kg source. The source is divided across active channels,
so four active drones do not each receive a separate free 5 kW transmitter.

Uniform beam, 0.10 m initial radius, 0.3 mrad half-angle, 0.25 m² receiver area,
conditional attenuation 0.0001/m, local dust coefficient 0.005, ideal pointing with
configurable receiver cosine. Receiver thermal model: 4000 J/K, passive 20 W/K,
25°C ambient, 70°C limit. These are **model assumptions**. Reference station mass
300 kg, input 5 kW and range 300 m are not verified ARC product specifications.
Published orientation: https://powerlighttech.com/wireless-power/ and
https://powerlighttech.com/ptrol-uas-demo/ .

## Boundaries of this version

Greedy scheduling and simple routes can spend much of a long mission waiting or
relocating. Group lift is quasi-static, without rope dynamics, wind control or a
full volumetric load envelope. Sand is a conservative heightfield, not a continuum
soil mechanics solver; dry loose sand, simplified supports and empirical power.
Container wear is counted without random rupture/repair. No battery degradation,
motor aerodynamics, real DEM ingestion, hydrology or water erosion.

Large solar-field siting has a reserved rectangular footprint and timed deployment;
the deployed large field stays stationary. There is no detailed rack/cable transport
or panel shading model. Inventory quantities and physical specifications are set
before a logistics run; arbitrary live hardware shipments are unsupported. Legacy
live editing is preserved. Configuration files restart a scenario; full simulation
checkpoints are not implemented.
