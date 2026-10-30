#pragma once
#include "core/diffpd_core.h"
#include "energy/energy_springs.h"
#include "energy/energy_arap.h"
#include "energy/energy_hinge.h"
#include "energy/energy_seam.h"


// ----------------
// ELASTIC ENERGIES
// ----------------

// Every elastic energy type is one struct owning all its elements (energy_springs.h is the template), so the
// per-element loop sits inside each method and there is no per-element dispatch. `pinned` is mesh.pinned_rest.
//   size_t triplet_count() const;                                          upper bound of one stiffness emission
//   void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const;    scale * K, K = sum_e w_e G_e^T G_e
//   void add_forces(x, pinned, RealVecX& b, Real scale) const;                  b += scale * sum_e w_e G_e^T p_e(x)
//   void precompute_local_derivative(x, pinned);                                caches Gamma_e, p_e at x = x+_t
//   void add_jacobian_product(v, RealVecX& out) const;                          out += sum_e G_e^T Gamma_e G_e v
//   void add_param_gradient(z_perp, x_minus, v_plus, pinned, h, h2_eff, Stiffness& grad) const;
// Ordering contract: precompute_local_derivative(x+_t) runs (in compute_adjoint_vector_contact) before the
// step's add_jacobian_product / add_param_gradient. Gamma_e must be symmetric (true for projections: ARAP's
// dR/dF is a Hessian), else add_jacobian_product must apply its transpose. Parameters live in a SpringKind slot.
//
// Adding an energy (e.g. ARAP): write energy_<name>.h with the methods above plus its own generation helper
// (like SpringEnergy::add_edge), add a member below and to both for_each bodies, call the helper where its
// elements are generated, and pick a SpringKind slot for its stiffness.
struct ElasticEnergies
{
    SpringEnergy springs;
    ArapEnergy   arap; // forward only
    HingeEnergy  bending; // forward only; generated with ARAP on the Tri skirt
    SeamEnergy   seams;   // fixed stiffness, no parameter gradient (energy_seam.h)

    template<class F> void for_each(F&& f)       { f(springs); f(arap); f(bending); f(seams); }
    template<class F> void for_each(F&& f) const { f(springs); f(arap); f(bending); f(seams); }
};
