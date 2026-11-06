#pragma once

// The contact impulse of the impulse-based (Ly et al.) contact method: the local step of pd_contact, and the same
// pressing test the backward pass repeats to decide which contacts are active.
//
// Two independent axes, kept apart:
//   MODEL (how the local datum d is built):  relative velocity | gap-aware   (d_relvel / d_gap, scalar d_n_*)
//   RULE  (what impulse d gives):            frictionless | Coulomb           (impulse_frictionless / impulse_coulomb)
// The rules are pure functions of d, n (and mu) and never know the model. contact_d_n / contact_impulse are the only
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
// take-off r = 0; stick r = -d (u = 0); slip r_T = -mu r_N d_T/|d_T|. Forward only (no backward counterpart yet).
inline Vec3 impulse_coulomb(const Vec3& d, const Vec3& n, Real mu)
{
    const Real d_n = d.dot(n);
    if (d_n >= 0.0) return Vec3::Zero();
    const Vec3 d_t = d - d_n * n;
    const Real r_n = -d_n;
    if (d_t.norm() <= mu * r_n) return -d;
    return r_n * n - mu * r_n * d_t.normalized();
}

// The impulse of contact `c`: model picks d, mu picks the rule (mu == 0 is the frictionless path, bit for bit the old one).
inline Vec3 contact_impulse(const Contact& c, bool gap, const Vec3& f_i, Real m_i, Real h, Real mu)
{
    if (mu == 0.0) return impulse_frictionless(contact_d_n(c, gap, f_i, m_i, h), c.normal);
    return impulse_coulomb(gap ? d_gap(f_i, m_i, c.v_c, c.normal, c.gap / h) : d_relvel(f_i, m_i, c.v_c), c.normal, mu);
}
