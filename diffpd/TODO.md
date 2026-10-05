# diffpd TODO / Notes

## Done

- [x] Pin vertices rendered bigger and yellow
- [x] Collider collision using projected closest surface point (`ContactPointMode::Surface`)
- [x] Support more than one collider at once
- [x] Add smoothness with subdivision to cloth in final simulation render

## Open

- [ ] Collider collision using some form of CCD (continuous collision detection)
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [ ] Cloth self-contact

- [ ] add substeps to animated colliders

- [ ] Try derive gradient correction for: contact_point_mode == ContactPointMode::Particle
- [ ] Try derive gradient correction for: animated_collider_velocity_mode == AnimatedColliderVelocityMode::DecomposedRigid
- [ ] Check gradient works correctly in the case of an animated collider being a Sphere

### Triangle mesh collider

- [ ] Add triangle mesh collider type (forward + backward)
  - [ ] Derive backward pieces for a triangle:
    - Curvature correction: probably not needed (flat, shape operator = 0, like Plane)
    - Rotation correction: still needed, must be derived
  - [ ] Performance: find a way to keep it fast (e.g. AABB pre-check per triangle / BVH)
  - [ ] Best-fit triangle selection: when a vertex is inside the mesh, which triangle is the actual contact?
    - Closest triangle?
    - Triangle with the shortest normal-projection distance?
