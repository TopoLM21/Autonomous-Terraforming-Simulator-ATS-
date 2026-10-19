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

`batteries.py` runs timed sequential side-dock swaps and base drone swaps. Each
replaceable physical module has a permanent `ChargingSlot` with its own controller
and ID. This includes initially mounted packs, so a removed pack always has a home
slot. Away/mounted packs leave empty slots; reserved packs cannot charge. Base-bank
batteries use their original storage circuit. Every slot adds its configured
hardware mass (model default 1 kg) to the base, with no extra stored energy.
Offboard packs can charge concurrently at up to 1.5 kW each; mounted drone banks
use the preserved aggregate 8 kW landing circuit, which has priority. Remaining
source power is shared equally across incomplete modules, redistributing unused
power when a module reaches full charge or its local limit. Both branches,
mechanics, electronics and optical links share finite generation/storage and the
100 kW inverter limit. `individual_chargers=False` preserves the previous finite
port queue and common 8 kW charge limit for historical comparisons. Mechanical docks
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

Idle carrier work sharing is enabled by `LogisticsConfig.idle_digging`. After
30 idle seconds a carrier may request its stored grabber; exchange takes 45 s
and 100 W at a real landing/service pad, with both tool IDs reserved. Ground
waiting positions are not service stations: aircraft fly to a free pad and
idle occupants yield occupied pads through an energy-budgeted flight. One hook
carrier is retained when several drones serve the excavator. Queued demand is matched against ready aircraft; charging carriers do not cover
waiting jobs. Available diggers may remount their hooks for cargo while the other
diggers work independently. Queued transport never globally disables excavation.
If the available diggers are airborne, one can finish its soil delivery and return
for a physical tool change. In strict hybrid mode, the dedicated digger is retained
while a non-charging hook carrier is available. Relocation recalls direct sorties. Charging and
mechanical battery reservations prevent concurrent dispatch. Direct and transport
jobs cannot own the same aircraft. Disable idle digging for strict historical
logistics comparisons; `direct_allowed` still controls initially dedicated diggers.

`Simulation.report(full=False)` computes live balances and fleet telemetry without
serializing inventory or historical jobs. Main GUI refresh uses this path; the
inventory dialog reads items only on its inventory tab, and the dispatcher slices
the last 200 jobs before formatting. Table cells update in place; user column widths
and scroll positions persist. The default report remains complete for CLI exports.
The modeless dialog is reused and its timer stops on close. Grounded drones report
the reason for waiting on the map and in their fleet-state tooltip. An unaffordable
direct sortie requests a full charging cycle even just above the low-SOC threshold;
this also applies to a preview of a heavier grabber before mounting it. A collective
lift requests charge only from deficient participants and never interrupts an
existing request before 100%. A foreign grabber reservation prevents direct launch. Mechanical drone swaps
have a grounded `SWAPPING_BATTERIES` state, pack IDs, remaining time, and per-aircraft
counters; removed packs return to their own base charging slots.

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
| Removed-pack slots / input limit | One per replaceable physical module / 1.5 kW each | Model, shared finite source |
| Slot hardware mass | 1 kg each | Model, included in base mass |
| Drone pack input / discharge rating | 4 / 60 kW | Compatibility assumptions for existing heavy flight model |
| Base bank charge / discharge rating | 2C / 10C | Model, also constrained by inverter |
| Base inverter / landing charge input | 100 / 8 kW | Model / existing dock charge input preserved |
| Common rig / cable angle / lift margin | 12 kg / 30° / 1.2 | Model |
| Container tare / volume / gross | 5 kg / 55 L / 90 kg | Model |
| Large field | 200 m², mass from panel recipes, 300 s deployment at 200 W | Separate stationary scenario |

Initial spare counts are exposed in the experiment dialog, `experiment_config`,
and CLI `--drone-spare-sets` / `--excavator-spare-batteries`. Drone reserves count
complete sets per aircraft; excavator reserves count individual offboard modules.
Explicit CLI counts override loaded configuration, omitted flags retain saved
values. Every additional pack has finite mass and initial energy, without a runtime
energy import. Base mass includes the warehouse before the first physical step.

All settings are in `LogisticsConfig` or existing equipment/configuration classes.
Internal states use seconds, metres, kilograms, joules and watts; user-facing
capacity inputs marked kWh are converted into joules on instantiation.

## Field extent and camera

Fresh CLI/GUI launches use a 64 m square; `--map-size` and new-scenario controls
accept 32–256 m at unchanged SI coordinates and cell size. With the original
0.25 m resolution this is 128–1024 cells per side. `Config()` and loaded files
retain their original extent unless explicitly overridden. `maps.with_map_size`
clones and validates the configuration, requires an integer cell count and refuses
to resize a stored height grid. Saved plans/configurations retain their field size.

Zoom buttons and the wheel change only the camera. Full-screen mode reparents the
same plot/frame to a dialog, keeping the simulation, brush and view state. F11/Esc
restore that frame; no second simulator is created. Main-window close also closes
the full-screen dialog. Inventory slot details are built only on their visible
tab, not during every main-window telemetry refresh.

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
