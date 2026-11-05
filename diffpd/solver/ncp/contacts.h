#pragma once

// Contact set of one NCP step: our detect_contacts / stamp_contacts, plus what only this method needs (tangent basis, the
// contact Jacobian H, the offsets d, mu). Frozen for the whole step, like the paper's DetectContacts.

#include "collision/detect.h"
#include "solver/ncp/settings.h"

// Orthonormal (t1, t2) with t1 x t2 = n: Gram-Schmidt from the axis least aligned with n.
inline void tangent_basis(const Vec3& n, Vec3& t1, Vec3& t2)
{
    Index k;
    n.cwiseAbs().minCoeff(&k);
    const Vec3 e = Vec3::Unit(k);
    t1 = (e - e.dot(n) * n).normalized();
    t2 = n.cross(t1);
}

struct NcpContacts
{
    Contacts   contacts;      // the base contacts (particle, normal, surface_point, v_c), one per particle; empty in unit tests
    SparseMat  H;             // 3m x 3n. Rows [0, m): normal n_j. Rows m + 2j, m + 2j + 1: tangents t1_j, t2_j of contact j
    RealVecX   d;             // 3m. delta = H x - d: normal rows d_n = n.s; tangent rows h t.v_c + t.x^- (x^- folded in, so the slip delta_f = H_f (x - x^-) - h t.v_c)
    Real       mu = 0.0;

    Index m() const { return d.size() / 3; }
};

// Contacts at x_tilde (end-of-step collider pose, closest surface point s and normal n from x_tilde, like the impulse method's
// RelativeVelocity contact), within `margin` of the surface. One contact per particle (detect_contacts claims a particle once).
inline NcpContacts detect_ncp_contacts(const CollisionWorld& world, const Positions& x_tilde, const StepClock& clock, int step,
                                       const Positions& x_minus, Real h, const NcpSettings& s)
{
    ASSERT(!world.contact.gap(), "NCP contact needs ContactModel::RelativeVelocity: the Gap model re-evaluates the geometry at x-");
    NcpContacts k;
    k.mu       = s.friction;
    k.contacts = detect_contacts(world, x_tilde, clock.contact_time(step), -s.margin);
    stamp_contacts(world, clock, step, x_minus, k.contacts, 0, k.contacts.size());

    const Index m = (Index)k.contacts.size();
    std::vector<Triplet> trip;
    trip.reserve(9 * m);
    k.d = RealVecX::Zero(3 * m);
    for (Index j = 0; j < m; ++j)
    {
        const Contact& c = k.contacts[j];
        Vec3 t1, t2;
        tangent_basis(c.normal, t1, t2);
        const Vec3 xm = x_minus.segment<3>(3 * c.particle);
        const Vec3 rows[3] = { c.normal, t1, t2 };
        const Index ids[3] = { j, m + 2 * j, m + 2 * j + 1 };
        for (int r = 0; r < 3; ++r)
            for (int a = 0; a < 3; ++a)
                trip.emplace_back(ids[r], 3 * c.particle + a, rows[r](a));
        k.d(j)             = c.normal.dot(c.surface_point);
        k.d(m + 2 * j)     = h * t1.dot(c.v_c) + t1.dot(xm);
        k.d(m + 2 * j + 1) = h * t2.dot(c.v_c) + t2.dot(xm);
    }
    k.H.resize(3 * m, x_tilde.size());
    k.H.setFromTriplets(trip.begin(), trip.end());
    return k;
}
