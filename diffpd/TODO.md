# diffpd TODO / Notes

## Done

- [x] Pin vertices rendered bigger and yellow
- [x] Collider collision using projected closest surface point (`ContactPointMode::Surface`)

## Open

- [ ] Collider collision using some form of CCD (continuous collision detection)
- [ ] Support more than one collider at once
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [ ] Cloth self-contact

## Notes: skeletal animation collider velocity

For when we add skeletal animation — getting instantaneous angular velocity out of a bone.

Skeletal animation gives **poses** (orientation quaternion/matrix per joint per frame), not velocity directly.

- **Finite difference** — simplest, matches baked animation: `ω` from the quaternion delta between consecutive frames.
- **Analytic** — if driven by explicit joint-angle rate functions `θ(t)`: `ω = dθ/dt · joint_axis`.
