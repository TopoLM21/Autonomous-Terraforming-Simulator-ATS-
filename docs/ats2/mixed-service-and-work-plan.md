# Mixed crews: service decks and a reusable work plan

The mixed site uses one terrain and material ledger. Buying the diesel fleet
also enables the shared obstacle geometry; it does not give the drones a
separate surface or change their battery charging policy.

## Landing on sloping ground

A landing pad belongs to a particular rigid carrier. Its deck height and the
collision roof now use the same carrier-centre ground sample. Previously the
pad sampled the ground below its offset position instead. On a 0.5% slope the
two values could differ by only 7.5 mm, but this was enough to put the final
landing target inside the carrier roof and block the autonomous mission.

The diesel fleet could continue advancing the common clock while that
autonomous mission was blocked. Consequently other drones appeared to remain
in `SWAPPING_BATTERIES`, `RETURN` or `LANDING` indefinitely. The interface now
retains the blocked complex's reason in the group status and shows
`Остановлен · <operation>` in the drone list, with the reason in its tooltip.
An actual obstruction still blocks movement; collision checks are retained.

Snapshot version 23 reads previous versions. On old saves it repairs only
aircraft exactly at their historical, owned dock anchor and landing targets
exactly equal to that anchor. Aircraft in flight are not repositioned and
battery energy is not changed. Unfinished powered module handling is replanned
from its saved physical position. Only the identified vertical landing
collision can be retried automatically; unrelated blocked reasons are kept.
New-version save/load does not repeat this migration.

## Reusable plan for the conventional fleet

The brush/channel already stores the target surface. The conventional fleet
additionally caches its work-cell coordinates, ramp projections and approach
stations. Current heights and ramp floors are read only on those work cells
when choosing the next cut. Geometry is rebuilt after changes to the work
area, ramp or active terrain, rather than on every assignment.

New conventional fleets fill the dump **from the far edge towards the fixed
fuel camp/access**, independently of the truck's changing position. Both that
order and the previous nearest-place policy are available in
`Управление → Экскаватор и самосвалы…`. The work tab displays the derived plan;
`Применить к новым рейсам` changes future assignments without cancelling cargo
or a committed route. Older saves retain their previous nearest-place policy.

Only geometric ordering is cached. Residual excavation, slumping, dump
capacity, occupied stances and the full swept exit route remain live checks.
A blocked candidate can be reconsidered when it becomes free. Existing route
search limits and the retained access ramp remain in force. Unloading reuses
the checked return route and validates its complete swept path before tipping;
a real change can still require a new search. Derived plan arrays are not
stored in snapshots and never constitute another soil balance.

Drone unloading continues to choose local, energy-budgeted points. Applying
the far-to-near rule to the entire shared dump would send drones to remote
banks, and the emergency rerouting of a loaded drone also calls the same
low-level allocator. A future drone ordering policy must therefore be bounded
to its working sector/bank and retain an energy-safe local fallback.

## Verification limits

The landing defect is reproduced with an actual mixed-fleet installation on a
synthetic 0.5% slope. This does not establish that every reported wait has the
same cause. A waiting drone should be diagnosed from the displayed mission
reason, battery/service state and reservations. Short regression runs do not
validate multi-day operation or the earlier reported 23-hour UI freeze.

The [validation record](mixed-service-validation.json) contains source hashes,
the bounded physical comparison and CLI checks. With the compatibility
nearest policy, the 1000-second control preserved the heightfield hash,
material amounts, cargo, volumes and fuel; route searches fell from 16 to 13.
Work/dump geometry each built once. This is a small headless control, not a
claim about the user's GUI speed. Far-to-near intentionally changes haul
distances and can cost more time or fuel than nearest-place dispatch.
