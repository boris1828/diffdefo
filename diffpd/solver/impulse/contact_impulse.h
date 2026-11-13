#pragma once

// The contact impulse of the impulse-based (Ly et al.) contact method: the local step of pd_contact, and the same
// pressing test the backward pass repeats to decide which contacts are active.
//
// Two independent axes, kept apart:
//   MODEL (how the local datum d is built):  relative velocity | gap-aware   (d_relvel / d_gap, scalar d_n_*)
//   RULE  (what impulse d gives):            frictionless | Coulomb           (impulse_frictionless / impulse_coulomb)
// The rules are pure functions of d, n (and mu) and never know the model; the Coulomb rule's branch is coulomb_state. contact_d_n / contact_d / contact_impulse are the only
// switch on the model; forward and backward both go through them, so their pressing test and their d are the same by construction.

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

// The whole vector d, contact_d_n's counterpart for the friction rule. Its normal part equals contact_d_n up to rounding
// (the scalar builders are not computed as d . n), so the frictionless path keeps its own scalar.
inline Vec3 contact_d(const Contact& c, bool gap, const Vec3& f_i, Real m_i, Real h)
{
    return gap ? d_gap(f_i, m_i, c.v_c, c.normal, c.gap / h) : d_relvel(f_i, m_i, c.v_c);
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

// take-off r = 0; stick r = -d (u = 0); slip r_T = -mu r_N d_T/|d_T|. Derivatives below (coulomb_dr_*); the backward does not use them yet.
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

// ---- RULE derivatives: Coulomb (for the friction backward, plans/FRICTION_BACKWARD_PLAN.md) ----
// Slip, with d_n = d.n, d_T = d - d_n n, s = |d_T| > 0 (coulomb_state guarantees it), t = d_T/s, kappa = mu r_n/s = -mu d_n/s:
//   r = -d_n n + mu d_n t
// Take-off (r = 0) and stick (r = -d) do not depend on n or mu. Each function branches on coulomb_state, like the rule.
// The model sits outside: relative velocity d = f - m v_c has dd/df = I, dd/dv_c = -m I, dd/dn = 0, so dr/df = R, dr/dv_c = -m R.

// R = dr/dd. Not symmetric in slip (mu t n^T: pressing harder lengthens the friction arrow): the adjoint needs R^T.
inline Mat3 coulomb_dr_dd(const Vec3& d, const Vec3& n, Real mu)
{
    const Real d_n = d.dot(n);
    const Vec3 d_t = d - d_n * n;
    const Real s   = d_t.norm();
    switch (coulomb_state(d_n, s, mu))
    {
    case CoulombState::TakeOff: return Mat3::Zero();
    case CoulombState::Stick:   return -Mat3::Identity();
    case CoulombState::Slip:    break;
    }
    const Vec3 t     = d_t / s;
    const Real kappa = -mu * d_n / s;
    const Mat3 nn    = n * n.transpose();
    return -nn + mu * t * n.transpose() - kappa * (Mat3::Identity() - nn - t * t.transpose());
}

// (dr/dn)^T z, projected onto the tangent plane of n: what a tilt of the normal does to r at fixed d, pulled back. The
// caller turns it into dphi/dx with the collider's shape operator. With n a free vector,
//   dr/dn = (kappa - 1) n d^T + mu t d^T - d_n I + kappa d_n (I - t t^T)
// and its normal component is dropped (n stays unit). At mu = 0: -z_n d_T - d_n z_perp, the frictionless term of normal_correction.
inline Vec3 coulomb_dr_dn_pullback(const Vec3& d, const Vec3& n, Real mu, const Vec3& z)
{
    const Real d_n = d.dot(n);
    const Vec3 d_t = d - d_n * n;
    const Real s   = d_t.norm();
    if (coulomb_state(d_n, s, mu) != CoulombState::Slip) return Vec3::Zero();
    const Vec3 t     = d_t / s;
    const Real kappa = -mu * d_n / s;
    const Mat3 J = (kappa - 1.0) * n * d.transpose() + mu * t * d.transpose()
                 - d_n * Mat3::Identity() + kappa * d_n * (Mat3::Identity() - t * t.transpose());
    const Vec3 g = J.transpose() * z;
    return g - n * n.dot(g);
}

// dr/dmu = d_n t = -r_n t in slip (a wider cone lengthens the friction arrow), 0 otherwise (stick was already held).
inline Vec3 coulomb_dr_dmu(const Vec3& d, const Vec3& n, Real mu)
{
    const Real d_n = d.dot(n);
    const Vec3 d_t = d - d_n * n;
    const Real s   = d_t.norm();
    if (coulomb_state(d_n, s, mu) != CoulombState::Slip) return Vec3::Zero();
    return (d_n / s) * d_t;
}

// The impulse of contact `c`: model picks d, mu picks the rule (mu == 0 is the frictionless path, bit for bit the old one).
inline Vec3 contact_impulse(const Contact& c, bool gap, const Vec3& f_i, Real m_i, Real h, Real mu)
{
    if (mu == 0.0) return impulse_frictionless(contact_d_n(c, gap, f_i, m_i, h), c.normal);
    return impulse_coulomb(contact_d(c, gap, f_i, m_i, h), c.normal, mu);
}
