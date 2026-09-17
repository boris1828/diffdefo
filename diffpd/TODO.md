# diffpd TODO / Notes

## Done

- [x] Pin vertices rendered bigger and yellow
- [x] Collider collision using projected closest surface point (`ContactPointMode::Surface`)
- [x] Support more than one collider at once

## Open

- [ ] Collider collision using some form of CCD (continuous collision detection)
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [ ] Cloth self-contact

- [ ] Try derive gradient correction for: contact_point_mode == ContactPointMode::Particle
- [ ] Try derive gradient correction for: animated_collider_velocity_mode == AnimatedColliderVelocityMode::DecomposedRigid
- [ ] Check gradient works correctly in the case of an animated collider being a Sphere