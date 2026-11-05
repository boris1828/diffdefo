#pragma once

// The contact impulse of the impulse-based (Ly et al.) contact method: the local step of pd_contact, and the same
// threshold test the backward pass repeats to decide which contacts are pressing.

#include "core/diffpd_core.h"

// d = f_i - m_i * v_c : the free-force target shifted by the obstacle's own translation
// velocity (Ly et al., "Moving obstacle", Eq. 10: d_j := R_j^T f_i + M_i u_f,j, u_f,j = -v_c).
// Speculative contact: v_c = 0 and gap_speed = phi(x-)/h, i.e. v+.n >= -phi(x-)/h.
// Pressing (impulse needed) when negative. One function for forward and backward so their parity is by construction.
inline Real contact_d_n(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n, Real gap_speed = 0.0)
{
    return (f_i - m_i * v_c).dot(n) + m_i * gap_speed;
}

inline Vec3 update_contact_force(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n, Real gap_speed = 0.0)
{
    const Real d_n = contact_d_n(f_i, m_i, v_c, n, gap_speed);
    if (d_n >= 0.0) return Vec3::Zero();
    return -d_n * n;
}

// Coulomb friction (Ly et al., sec. 3.3): same d as above; take-off r = 0, stick r = -d (u = 0), slip r_T = -mu r_N d_T/|d_T|.
// Forward only (no backward counterpart yet) and no gap term: pd_contact asserts both.
inline Vec3 update_contact_force_friction(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n, Real mu)
{
    const Vec3 d   = f_i - m_i * v_c;
    const Real d_n = d.dot(n);
    if (d_n >= 0.0) return Vec3::Zero();
    const Vec3 d_t = d - d_n * n;
    const Real r_n = -d_n;
    if (d_t.norm() <= mu * r_n) return -d;
    return r_n * n - mu * r_n * d_t.normalized();
}
