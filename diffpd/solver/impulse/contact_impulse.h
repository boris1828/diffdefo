#pragma once

// The contact impulse of the impulse-based (Ly et al.) contact method: the local step of pd_contact, and the same
// pressing test the backward pass repeats to decide which contacts are active.
//
// Two independent axes, kept apart:
//   MODEL (how the local datum d is built):  relative velocity | gap-aware   (d_relvel / d_gap, scalar d_n_*)
//   RULE  (what impulse d gives):            frictionless | Coulomb           (impulse_frictionless / impulse_coulomb)
// The rules are pure functions of d, n (and mu) and never know the model; the Coulomb rule's branch is coulomb_state. contact_d_n / contact_impulse are the only
// switch on the model; forward and backward both go through contact_d_n, so their pressing test is the same by construction.

#include "collision/world.h"
#include "core/diffpd_core.h"

// ---- MODEL: relative velocity ----
// d = f_i - m_i v_c: the free-force target shifted by the obstacle's own velocity (Ly et al., "Moving obstacle", Eq. 10:
// d_j := R_j^T f_i + M_i u_f,j, u_f,j = -v_c). Contact is at x~, constraint (v+ - v_c).n >= 0.
inline Vec3 d_relvel(const Vec3& f_i, Real m_i, const Vec3& v_c) { return f_i - m_i * v_c; }
inline Real d_n_relvel(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n) { return (f_i - m_i * v_c).dot(n); }

// ---- MODEL: gap-aware (speculative) ----
// Contact at x-, constraint phi(x-)/h + v+.n >= 0 (phi_h = phi/h): d_n = f.n + m phi/h. The collider's normal motion is
// already inside phi, so of the obstacle velocity v_c only its tangential part (friction) enters d: it is projected here.
inline Vec3 d_gap(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n, Real phi_h)
{
    return f_i - m_i * (v_c - v_c.dot(n) * n) + (m_i * phi_h) * n;
}
inline Real d_n_gap(const Vec3& f_i, Real m_i, const Vec3& n, Real phi_h) { return f_i.dot(n) + m_i * phi_h; }

// Pressing (impulse needed) when negative. `gap` = world.contact.gap(), `h` the step size.
inline Real contact_d_n(const Contact& c, bool gap, const Vec3& f_i, Real m_i, Real h)
{
    return gap ? d_n_gap(f_i, m_i, c.normal, c.gap / h) : d_n_relvel(f_i, m_i, c.v_c, c.normal);
}

// ---- RULE: frictionless ----
inline Vec3 impulse_frictionless(Real d_n, const Vec3& n)
{
    if (d_n >= 0.0) return Vec3::Zero();
    return -d_n * n;
}

// ---- RULE: Coulomb friction (Ly et al., sec. 3.3) ----
// The three branches of the rule. The impulse is continuous across them, its derivative dr/dd is not, so the backward
// pass must take the branch the forward took: both branch on coulomb_state, never on a test of their own.
enum class CoulombState : unsigned char
{
    TakeOff, // d_n >= 0: separating, r = 0
    Stick,   // d_n < 0 and |d_T| <= mu r_n (inside the cone, boundary included): r = -d, no relative motion
    Slip,    // d_n < 0 and |d_T| >  mu r_n: r = r_n n - mu r_n d_T/|d_T|, on the cone
};

// From the parts of d (d_n = d.n, d_t_norm = |d - d_n n|). Pitfalls:
//  - the boundary |d_T| = mu r_n is Stick (r is the same from both sides, R = dr/dd is not);
//  - d_T = 0 is Stick for every mu, even mu = 0 (0 <= 0): a node pressed straight onto a plane sticks, R = -I there,
//    not the frictionless -n n^T. Slip therefore always has d_T != 0, so d_T/|d_T| is safe in that branch;
//  - a NaN d gives Slip (both tests false), and the impulse then stays NaN, as before this split.
inline CoulombState coulomb_state(Real d_n, Real d_t_norm, Real mu)
{
    if (d_n >= 0.0) return CoulombState::TakeOff;
    if (d_t_norm <= mu * (-d_n)) return CoulombState::Stick;
    return CoulombState::Slip;
}
inline CoulombState coulomb_state(const Vec3& d, const Vec3& n, Real mu)
{
    const Real d_n = d.dot(n);
    return coulomb_state(d_n, (d - d_n * n).norm(), mu);
}

// take-off r = 0; stick r = -d (u = 0); slip r_T = -mu r_N d_T/|d_T|. Forward only (no backward counterpart yet).
inline Vec3 impulse_coulomb(const Vec3& d, const Vec3& n, Real mu)
{
    const Real d_n = d.dot(n);
    const Vec3 d_t = d - d_n * n;
    switch (coulomb_state(d_n, d_t.norm(), mu))
    {
    case CoulombState::TakeOff: return Vec3::Zero();
    case CoulombState::Stick:   return -d;
    case CoulombState::Slip:    break;
    }
    const Real r_n = -d_n;
    return r_n * n - mu * r_n * d_t.normalized();
}

// The impulse of contact `c`: model picks d, mu picks the rule (mu == 0 is the frictionless path, bit for bit the old one).
inline Vec3 contact_impulse(const Contact& c, bool gap, const Vec3& f_i, Real m_i, Real h, Real mu)
{
    if (mu == 0.0) return impulse_frictionless(contact_d_n(c, gap, f_i, m_i, h), c.normal);
    return impulse_coulomb(gap ? d_gap(f_i, m_i, c.v_c, c.normal, c.gap / h) : d_relvel(f_i, m_i, c.v_c), c.normal, mu);
}
