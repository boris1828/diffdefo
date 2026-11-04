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
