#include "diffpd_types.h"
#include "diffpd_viewer.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <functional>

namespace fs = std::filesystem;

template <typename ParamGrad>
struct BackwardGrad
{
    RealVecX  free_grad;  // dphi/dx for free particles (per-step: dx_minus; full: dx0)
    ParamGrad param_grad; // dphi/dtheta — shape determined by ParamGrad
};

std::string ANIM_DIR;

// ----------------
//      FILE
// ----------------

void clear_folder(const std::string& folder)
{
    namespace fs = std::filesystem;

    if (!fs::exists(folder))
    {
        fs::create_directories(folder);
        return;
    }

    ASSERT(fs::is_directory(folder), "clear_folder: not a directory: " << folder);

    for (const auto& entry : fs::directory_iterator(folder))
        if (entry.is_regular_file() && entry.path().extension() == ".obj")
            fs::remove(entry.path());
}

// ----------------
//    CONTACT
// ----------------

Collider collider = make_sphere(Vec3(0.0, -10.0, 0.0), 1.0); // default; overwritten in main()

Contacts detect_contacts(const Object& obj, const Positions& x, Real time)
{
    Contacts contacts;

    const Vec3 offset_t = collider.velocity * time;

    for (Index i = 0; i < obj.num_particles(); ++i)
    {
        const Vec3 pos = x.segment<3>(3*i);

        if (collider.type == ColliderType::Sphere)
        {
            const Vec3 offset           = pos - (collider.sphere_center + offset_t);
            const Real dist_from_center = offset.norm();
            const Real dist             = dist_from_center - collider.sphere_radius;
            if (dist < 0.0)
            {
                const Vec3 normal = offset / dist_from_center;
                contacts.push_back({static_cast<ParticleId>(i), normal, 1.0 / dist_from_center, false, 0.0});
            }
        }
        else if (collider.type == ColliderType::Cylinder)
        {
            const Vec3 axis    = collider.cylinder_axis.normalized();
            const Vec3 rel     = pos - (collider.cylinder_origin + offset_t);
            const Vec3 perp    = rel - rel.dot(axis) * axis;
            const Real rho     = perp.norm();
            const Real dist    = rho - collider.cylinder_radius;
            if (dist < 0.0)
            {
                const Vec3 normal = perp / rho;
                Contact c{static_cast<ParticleId>(i), normal, 1.0 / rho, false, 0.0};
                c.axis = axis;
                contacts.push_back(c);
            }
        }
        else if (collider.type == ColliderType::Plane)
        {
            const Vec3 normal = collider.plane_normal.normalized();
            const Real dist   = (pos - (collider.plane_origin + offset_t)).dot(normal);
            if (dist < 0.0)
                contacts.push_back({static_cast<ParticleId>(i), normal, 0.0, false, 0.0});
        }
        else // ColliderType::Capsule
        {
            const Vec3 p0 = collider.capsule_p0 + offset_t;
            const Vec3 p1 = collider.capsule_p1 + offset_t;
            const Vec3 axis_vec = p1 - p0;
            const Real L  = axis_vec.norm();
            const Vec3 a  = axis_vec / L;
            Real t = a.dot(pos - p0);
            t = std::clamp(t, 0.0, L);
            const Vec3 closest = p0 + t * a;
            const Vec3 offset  = pos - closest;
            const Real dist_from_axis = offset.norm();
            const Real dist = dist_from_axis - collider.capsule_radius;
            if (dist < 0.0)
            {
                const Vec3 normal = offset / dist_from_axis;
                Contact c{static_cast<ParticleId>(i), normal, 1.0 / dist_from_axis, false, 0.0};
                c.axis = (t > 0.0 && t < L) ? a : Vec3::Zero(); // cylinder regime vs. sphere-cap regime
                contacts.push_back(c);
            }
        }
    }

    return contacts;
}

// d = f_i - m_i * v_c : the free-force target shifted by the obstacle's own translation
// velocity (Ly et al., "Moving obstacle", Eq. 10: d_j := R_j^T f_i + M_i u_f,j, u_f,j = -v_c).
Vec3 update_contact_force(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n)
{
    const Real d_n = (f_i - m_i * v_c).dot(n);
    if (d_n >= 0.0) return Vec3::Zero();
    return -d_n * n;
}

// ----------------
//       LOSS
// ----------------

struct Loss
{
    Real                  total;
    std::vector<RealVecX> dloss_dx;

    Loss(const Tape& guess, const Tape& target, int sample_every)
    {
        ASSERT(guess.positions.size() == target.positions.size(),
               "tape size mismatch: " << guess.positions.size() << " vs " << target.positions.size());
        ASSERT(sample_every > 0, "sample_every must be positive");

        const int   n_frames = (int)guess.positions.size();
        const int   n_steps  = n_frames - 1;
        const Index dofs     = 3 * guess.positions[0].rows();

        total = 0.0;
        dloss_dx.resize(n_frames, RealVecX::Zero(dofs));

        for (int t = 0; t < n_frames; ++t)
        {
            const bool sampled = (t == n_steps) || ((n_steps - t) % sample_every == 0);
            if (!sampled) continue;

            const RealVecX xg = Eigen::Map<const RealVecX>(guess.positions[t].data(),  dofs);
            const RealVecX xt = Eigen::Map<const RealVecX>(target.positions[t].data(), dofs);

            const RealVecX diff = xg - xt;
            total       += diff.squaredNorm() / Real(dofs);
            dloss_dx[t]  = (2.0 / Real(dofs)) * diff;
        }
    }
};

// ----------------
//      CLOTH
// ----------------

// PinMode, HangingMode, ClothFlags, and cloth()'s declaration (with defaults) now live in
// diffpd_types.h, so diffpd_viewer.cpp can call cloth() to rebuild the live config-screen preview.

constexpr Real cloth_size = 2.0;

Object cloth(
    Index       width,
    Index       height,
    Real        stiffness,
    Vec3        origin,
    PinMode     pin_mode,
    HangingMode hanging_mode,
    uint8_t     flags,
    Real        m_tot)
{
    ASSERT(flags & ClothFlags::STRETCH, "cloth must have stretch constraints enabled");

    const Index N  = width * height;
    const Real  sx = cloth_size / Real(width  - 1);
    const Real  sz = cloth_size / Real(height - 1);

    auto grid = [height](Index i, Index j) { return i * height + j; };

    // HORIZONTAL: (i,j) -> (x, 0, z), flat, falls/swings under gravity.
    // VERTICAL:   (i,j) -> (x, -z, 0), i.e. a -90 deg rotation about X (y,z)->(-z,y) with y≡0,
    //             so the j=0 row (pinned by PinMode::ROW) stays at the top and hangs straight down.
    std::vector<Vec3> pos(N);
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
            pos[grid(i, j)] = origin + (hanging_mode == HangingMode::VERTICAL
                ? Vec3(i * sx, -j * sz, 0.0)
                : Vec3(i * sx, 0.0, j * sz));

    // --- mark pinned vertices ---------------------------------------------
    std::vector<bool> pinned(N, false);
    switch (pin_mode) {
        case PinMode::NONE:
            break;
        case PinMode::CORNERS:
            pinned[grid(0, 0)]         = true;
            pinned[grid(width - 1, 0)] = true;
            break;
        case PinMode::ROW:
            for (Index i = 0; i < width; ++i)
                pinned[grid(i, 0)] = true;
            break;
    }

    // --- compact free vertices into DOF indices (V_free, DOF_map) ---------
    // dof[v] = particle id in the stacked state, or -1 if pinned.
    std::vector<ParticleId> dof(N, -1);
    Index n = 0; // |V_free|
    for (Index v = 0; v < N; ++v)
        if (!pinned[v]) dof[v] = ParticleId(n++);

    Object obj;

    // --- state vectors (stacked 3n, free particles only) ------------------
    obj.x.resize(3 * n);
    for (Index v = 0; v < N; ++v)
        if (dof[v] >= 0)
            obj.x.segment<3>(3 * dof[v]) = pos[v];

    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    // --- mass: m_tot distributed uniformly over n free particles ----------
    obj.mass = MassDiag::Constant(3 * n, m_tot / Real(n));

    // ---- bake export mesh (after dof[] and pos are built) ------------------
    SimMesh& mesh = obj.mesh;

    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
        {
            const Index v = grid(i, j);
            SimMesh::Vertex mv;
            if (dof[v] >= 0)
            {
                mv.dof = dof[v];
            }
            else
            {
                mv.dof = ParticleId(-Index(mesh.pinned_rest.size()) - 1);
                mesh.pinned_rest.push_back(pos[v]);
            }
            mesh.vertices.push_back(mv);
        }

    // structural edges only (both endpoints, pinned or not)
    auto mesh_edge = [&](Index a, Index b)
    {
        mesh.edges.emplace_back(a, b);
    };

    for (Index i = 0; i < width - 1; ++i)
        for (Index j = 0; j < height; ++j)
            mesh_edge(grid(i, j), grid(i + 1, j));
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height - 1; ++j)
            mesh_edge(grid(i, j), grid(i, j + 1));

    // --- constraint emission ----------------------------------------------
    auto emit = [&](Index a, Index b) 
    {
        const Real l  = (pos[a] - pos[b]).norm();
        const bool pa = pinned[a], pb = pinned[b];
        if (pa && pb) return;
        if (!pa && !pb)
        {
            obj.constraints.push_back(
                Constraint::makeSpring2(stiffness, l, dof[a], dof[b]));
        }
        else
        {
            const Index  free   = pa ? b : a;
            const Index  anchor = pa ? a : b;
            const Vec3   xbar   = pos[anchor];
            obj.constraints.push_back(
                Constraint::makeSpring1(stiffness, l, dof[free], xbar));
        }
    };

    if (flags & ClothFlags::STRETCH) 
    {
        // structural: along width axis (i)
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 1, j));

        // structural: along height axis (j)
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 1; ++j)
                emit(grid(i, j), grid(i, j + 1));
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height - 1; ++j) 
            {
                emit(grid(i, j),     grid(i + 1, j + 1));
                emit(grid(i + 1, j), grid(i, j + 1));
            }
    }

    if (flags & ClothFlags::BENDING) 
    {
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 2; ++j)
                emit(grid(i, j), grid(i, j + 2));

        for (Index i = 0; i < width - 2; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 2, j));
    }

    return obj;
}

// ----------------
//   OBJ OUTPUT
// ----------------

void write_obj(const std::string& path, const Object& obj)
{
    const SimMesh& mesh = obj.mesh;

    std::ofstream out(path);
    ASSERT(out.is_open(), "could not open " << path);
    out << std::fixed << std::setprecision(6);

    for (Index vi = 0; vi < Index(mesh.vertices.size()); ++vi)
    {
        const Vec3 p = mesh.position(obj, vi);
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }
    for (const auto& e : mesh.edges)
        out << "l " << e.first + 1 << ' ' << e.second + 1 << '\n';
}

void write_obj_constraints(const std::string& path, const Object& obj)
{
    std::ofstream out(path);
    ASSERT(out.is_open(), "could not open " << path);
    out << std::fixed << std::setprecision(6);

    const Index n = obj.num_particles();

    for (Index i = 0; i < n; ++i)
    {
        const Vec3 p = obj.x.segment<3>(3 * i);
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }

    for (const Constraint& c : obj.constraints)
        if (c.type == SpringType::Spring1)
            out << "v " << c.spring1.xbar[0] << ' '
                        << c.spring1.xbar[1] << ' '
                        << c.spring1.xbar[2] << '\n';

    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            out << "l " << i1 + 1 << ' ' << i2 + 1 << '\n';
        }
    }

    Index anchor_vid = n + 1;
    for (const Constraint& c : obj.constraints)
        if (c.type == SpringType::Spring1)
            out << "l " << c.spring1.i + 1 << ' ' << anchor_vid++ << '\n';
}

void write_obj_frame(const Object& obj, int step, const std::string& prefix = "frame") 
{
    std::ostringstream name;
    name << prefix << "_" << std::setfill('0') << std::setw(6) << step << ".obj";
    write_obj((fs::path(ANIM_DIR) / name.str()).string(), obj);
}

// ----------------
//      SOLVER
// ----------------

void construct_lhs(Object& obj, Real dt)
{
    const Index n3 = obj.num_dofs();
    const Real  h2 = dt * dt;

    std::vector<Triplet> triplets;
    triplets.reserve(n3 + 12 * Index(obj.constraints.size()));

    // M / h²
    for (Index i = 0; i < n3; ++i)
        triplets.emplace_back(i, i, obj.mass(i) / h2);

    // sum_i k_i G_i^T G_i
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            for (int d = 0; d < 3; ++d)
            {
                triplets.emplace_back(3*i1+d, 3*i1+d, +c.k);
                triplets.emplace_back(3*i2+d, 3*i2+d, +c.k);
                triplets.emplace_back(3*i1+d, 3*i2+d, -c.k);
                triplets.emplace_back(3*i2+d, 3*i1+d, -c.k);
            }
        }
        else // SpringType::Spring1
        {
            const Index i = c.spring1.i;
            for (int d = 0; d < 3; ++d)
                triplets.emplace_back(3*i+d, 3*i+d, +c.k);
        }
    }

    obj.L.resize(n3, n3);
    obj.L.setFromTriplets(triplets.begin(), triplets.end());

    obj.solver = std::make_unique<Cholesky>();
    obj.solver->compute(obj.L);
    ASSERT(obj.solver->info() == Eigen::Success, "Cholesky factorization of L failed");
}

// Adds each constraint's elastic force contribution to `b` in place, scaled by `scale`.
// No allocation: `b` is the caller's buffer, sized to obj.num_dofs().
void add_elastic_forces(const Object& obj, const RealVecX& x, RealVecX& b, Real scale = 1.0)
{
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 e   = x.segment<3>(3*i1) - x.segment<3>(3*i2);
            const Vec3 p   = c.l * e / e.norm();
            b.segment<3>(3*i1) += scale * c.k * p;
            b.segment<3>(3*i2) -= scale * c.k * p;
        }
        else // SpringType::Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;
            const Vec3 e  = x.segment<3>(3*i) - xbar;
            const Vec3 p  = c.l * e / e.norm();
            b.segment<3>(3*i) += scale * c.k * (xbar + p);
        }
    }
}

// ----------------
// VELOCITY SOLVER
// ----------------

void construct_velocity_lhs(Object& obj, Real dt)
{
    const Index n3 = obj.num_dofs();
    const Real  h2 = dt * dt;

    std::vector<Triplet> triplets_L, triplets_C;
    triplets_L.reserve(n3 + 12 * Index(obj.constraints.size()));
    triplets_C.reserve(     12 * Index(obj.constraints.size()));

    // M (mass diagonal, not divided by h²)
    for (Index i = 0; i < n3; ++i)
        triplets_L.emplace_back(i, i, obj.mass(i));

    // h² · K  (elastic contribution)
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            for (int d = 0; d < 3; ++d)
            {
                triplets_L.emplace_back(3*i1+d, 3*i1+d, +h2 * c.k);
                triplets_L.emplace_back(3*i2+d, 3*i2+d, +h2 * c.k);
                triplets_L.emplace_back(3*i1+d, 3*i2+d, -h2 * c.k);
                triplets_L.emplace_back(3*i2+d, 3*i1+d, -h2 * c.k);

                triplets_C.emplace_back(3*i1+d, 3*i1+d, +h2 * c.k);
                triplets_C.emplace_back(3*i2+d, 3*i2+d, +h2 * c.k);
                triplets_C.emplace_back(3*i1+d, 3*i2+d, -h2 * c.k);
                triplets_C.emplace_back(3*i2+d, 3*i1+d, -h2 * c.k);
            }
        }
        else // Spring1
        {
            const Index i = c.spring1.i;
            for (int d = 0; d < 3; ++d)
            {
                triplets_L.emplace_back(3*i+d, 3*i+d, +h2 * c.k);
                triplets_C.emplace_back(3*i+d, 3*i+d, +h2 * c.k);
            }
        }
    }

    obj.L.resize(n3, n3);
    obj.L.setFromTriplets(triplets_L.begin(), triplets_L.end());

    obj.C.resize(n3, n3);
    obj.C.setFromTriplets(triplets_C.begin(), triplets_C.end());

    obj.solver = std::make_unique<Cholesky>();
    obj.solver->compute(obj.L);
    ASSERT(obj.solver->info() == Eigen::Success, "Cholesky factorization of velocity LHS failed");
}

RealVecX construct_velocity_rhs(
    const Object&    obj,
    const RealVecX&  b_inertia,
    const RealVecX&  x_k,
    const Positions& x_minus,
    Real             h)
{
    const Real h2 = h * h;
    RealVecX   b  = b_inertia;
    add_elastic_forces(obj, x_k, b, h2);
    return (1.0 / h) * (b - obj.L * x_minus);
}

// ----------------
//    BACKWARD
// ----------------

void precompute_constraints_local_derivative(Object& obj, const Positions& x)
{
    for (Constraint& c : obj.constraints)
    {
        Vec3 e;

        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            e = x.segment<3>(3*i1) - x.segment<3>(3*i2);
        }
        else // SpringType::Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;
            e = x.segment<3>(3*i) - xbar;
        }

        const Real e_norm = e.norm();

        if (e_norm < 1e-9)
        {
            c.gamma  = Mat3::Zero();
            c.p_star = Vec3::Zero();
            c.e      = Vec3::Zero();
            continue;
        }

        const Vec3 e_hat  = e / e_norm;
        const Mat3 P_perp = Mat3::Identity() - (e_hat * e_hat.transpose());

        c.e      = e;
        c.p_star = c.l * e_hat;
        c.gamma  = (c.k * c.l / e_norm) * P_perp;
    }
}

// Applies the assembled local spring Jacobian ΔA (Σ_i Gᵢᵀ Γᵢ Gᵢ) to a vector — the same
// per-constraint pattern used by both the position- and velocity-space adjoint RHS.
RealVecX apply_spring_jacobian(const Object& obj, const RealVecX& v)
{
    RealVecX out = RealVecX::Zero(v.rows());

    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 d   = c.gamma * (v.segment<3>(3*i1) - v.segment<3>(3*i2));
            out.segment<3>(3*i1) += d;
            out.segment<3>(3*i2) -= d;
        }
        else // SpringType::Spring1
        {
            const Index i = c.spring1.i;
            out.segment<3>(3*i) += c.gamma * v.segment<3>(3*i);
        }
    }

    return out;
}

RealVecX construct_backward_rhs(
    const Object&   obj,
    const RealVecX& z,
    const RealVecX& dloss_dx,
    const RealVecX& dloss_dx_t)
{
    return apply_spring_jacobian(obj, z) + dloss_dx + dloss_dx_t;
}

void precompute_contacts_local_derivative(
    Contacts&          contacts,
    const Object&      obj,
    const Positions&   x_plus,
    const Velocities&  v_plus,
    const Positions&   x_minus,
    const Velocities&  v_minus,
    const Vec3&        gravity,
    Real               h)
{
    RealVecX x_tilde = x_minus + h * v_minus;
    const Vec3 dg    = h * h * gravity;
    for (Index i = 0; i < obj.num_particles(); ++i)
        x_tilde.segment<3>(3*i) += dg;

    const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde);
    const RealVecX b_tilde   = construct_velocity_rhs(obj, b_inertia, x_plus, x_minus, h);
    const RealVecX f         = b_tilde - obj.C * v_plus;

    for (Contact& c : contacts)
    {
        const Vec3 f_i = f.segment<3>(3 * c.particle);
        const Real m_i = obj.mass(3 * c.particle);
        const Real d_n = (f_i - m_i * collider.velocity).dot(c.normal);
        c.active = (d_n < 0.0);
        c.d_n    = d_n;
    }
}

struct ContactSplit
{
    RealVecX z_perp; // (I-P) z : normal component removed at active contacts
    RealVecX Pz;     // P z     : only the normal component, at active contacts
};

// Splits a vector into its active-contact-normal and perpendicular parts:
// v = (I-P) v + P v, where P = block-diag(n nᵀ) over active contacts.
ContactSplit split_by_contact(const Contacts& contacts, const RealVecX& v)
{
    ContactSplit s{ v, RealVecX::Zero(v.rows()) };
    for (const Contact& c : contacts)
    {
        if (!c.active) continue;
        const Vec3 vi = v.segment<3>(3 * c.particle);
        const Vec3 pn = c.normal * c.normal.dot(vi);
        s.Pz.segment<3>(3 * c.particle)     = pn;
        s.z_perp.segment<3>(3 * c.particle) = vi - pn;
    }
    return s;
}

RealVecX construct_backward_contact_rhs(
    const Object&   obj,
    const Contacts& contacts,
    const RealVecX& z,
    const RealVecX& dloss_dv,
    const RealVecX& dloss_dv_t,
    Real            h)
{

    const Real h2 = h * h;
    const ContactSplit split = split_by_contact(contacts, z);

    RealVecX b = h2 * apply_spring_jacobian(obj, split.z_perp);
    if (!contacts.empty())
        b += obj.C * split.Pz;

    return b + dloss_dv + dloss_dv_t;
}

RealVecX compute_adjoint_vector_contact(
    Object&           obj,
    Contacts&         contacts,
    const Positions&  x_plus,
    const Velocities& v_plus,
    const Positions&  x_minus,
    const Velocities& v_minus,
    const RealVecX&   dloss_dv,
    const RealVecX&   dloss_dv_t,
    const Vec3&       gravity,
    Real              h,
    int               n_iters_adjoint)
{
    precompute_constraints_local_derivative(obj, x_plus);
    precompute_contacts_local_derivative(contacts, obj, x_plus, v_plus, x_minus, v_minus, gravity, h);

    constexpr Real kConvergenceTol = 1e-4;

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    Real rel_residual = 0.0;
    for (int k = 0; k < n_iters_adjoint; ++k)
    {
        const RealVecX b_back = construct_backward_contact_rhs(obj, contacts, z, dloss_dv, dloss_dv_t, h);
        const RealVecX z_new  = obj.solver->solve(b_back);
        rel_residual = (z_new - z).norm() / std::max(z_new.norm(), Real(1e-12));
        z = z_new;
    }
    if (n_iters_adjoint > 0 && rel_residual > kConvergenceTol)
    {
        std::cout << "  iter " << n_iters_adjoint << " rel_delta = " << rel_residual << "\n";
    }

    return z;
}

RealVecX compute_adjoint_vector(
    Object&          obj,
    const Positions& x_plus,
    const RealVecX&  dloss_dx,
    const RealVecX&  dloss_dx_t,
    int              n_iters_adjoint)
{
    precompute_constraints_local_derivative(obj, x_plus);

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    for (int k = 0; k < n_iters_adjoint; ++k)
    {
        const RealVecX b_back = construct_backward_rhs(obj, z, dloss_dx, dloss_dx_t);
        z = obj.solver->solve(b_back);
    }
    return z;
}

Vec3 compute_gradient_pinned_vertices(const Object& obj, const RealVecX& z)
{
    Vec3 grad = Vec3::Zero();
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring1)
        {
            const Vec3 zi = z.segment<3>(3 * c.spring1.i);
            grad += c.k * zi - c.gamma * zi; // (k·I − Γ)·z_i
        }
    }
    return grad;
}

Real compute_gradient_stiffness(const Object& obj, const RealVecX& z)
{
    Real grad = 0;
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Vec3 zi1 = z.segment<3>(3 * c.spring2.i1);
            const Vec3 zi2 = z.segment<3>(3 * c.spring2.i2);
            grad += (zi1 - zi2).dot(c.p_star - c.e);
        }
        else // Spring1
        {
            const Vec3 zi = z.segment<3>(3 * c.spring1.i);
            grad += zi.dot(c.p_star - c.e);
        }
    }
    return grad;
}

Real compute_gradient_stiffness_contact(
    const Object&     obj,
    const Contacts&   contacts,
    const RealVecX&   z,
    const Positions&  x_minus,
    const Velocities& v_plus,
    Real              h)
{
    const RealVecX z_perp = split_by_contact(contacts, z).z_perp;
    const Real     h2     = h * h;

    Real grad = 0;
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;

            const Vec3 e_minus = x_minus.segment<3>(3*i1) - x_minus.segment<3>(3*i2);
            const Vec3 dv      = v_plus.segment<3>(3*i1)  - v_plus.segment<3>(3*i2);
            const Vec3 dz      = z_perp.segment<3>(3*i1)  - z_perp.segment<3>(3*i2);

            grad += dz.dot(h * (c.p_star - e_minus) - h2 * dv);
        }
        else // Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;

            const Vec3 e_minus = x_minus.segment<3>(3*i) - xbar;
            const Vec3 vi      = v_plus.segment<3>(3*i);
            const Vec3 zi      = z_perp.segment<3>(3*i);

            grad += zi.dot(h * (c.p_star - e_minus) - h2 * vi);
        }
    }
    return grad;
}

struct AdjointStep
{
    RealVecX free_grad;
    RealVecX z;
};

AdjointStep backward_pd_step(
    Object&          obj,
    const Positions& x_plus,
    const RealVecX&  dloss_dx,
    const RealVecX&  dloss_dx_t,
    Real             dt,
    int              n_iters_adjoint)
{
    RealVecX z = compute_adjoint_vector(obj, x_plus, dloss_dx, dloss_dx_t, n_iters_adjoint);
    RealVecX free_grad = obj.mass.cwiseProduct(z) / (dt * dt);
    return { std::move(free_grad), std::move(z) };
}

// ----------------
//       PD
// ----------------

void pd_step(Object& obj, Real dt, const Vec3& gravity, int n_iters)
{
    const Real h2 = dt * dt;

    obj.prev_x = obj.x;
    RealVecX x_tilde = obj.x + dt * obj.v;
    const Vec3 dg = h2 * gravity;
    for (Index i = 0; i < obj.num_particles(); ++i)
        x_tilde.segment<3>(3*i) += dg;

    const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde) / h2;

    obj.x = x_tilde;

    RealVecX b(obj.num_dofs()); // reused every iteration below; same size each time, so no realloc
    for (int k = 0; k < n_iters; ++k)
    {
        b = b_inertia;
        add_elastic_forces(obj, obj.x, b);
        obj.x = obj.solver->solve(b);
    }

    obj.v = (obj.x - obj.prev_x) / dt;
}

void init_pd(Object& obj, Real dt)
{
    construct_lhs(obj, dt);
}

void init_pd_velocity(Object& obj, Real dt)
{
    construct_velocity_lhs(obj, dt);
}

void pd(Object& obj, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix)
{
    tape.clear();
    tape.record(obj);
    write_obj_frame(obj, 0);

    for (int step = 0; step < n_steps; ++step)
    {
        pd_step(obj, dt, gravity, n_iters);
        tape.record(obj);
        if (step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);

        if (step % 10 == 0) std::cout << "step " << step << "/" << n_steps << "\n";
    }
}

// Wang, "A Chebyshev Semi-Iterative Approach for Accelerating Projective and
// Position-Based Dynamics" (2015). Extrapolates each local/global iterate against the
// one two steps back; a pure convergence-speed wrapper, never changes what's solved.
// Default-constructed (enabled=false) reproduces the un-accelerated iteration exactly.
struct ChebyshevAccel
{
    bool enabled = false;
    Real rho     = 0.9992; // estimated spectral radius of the iteration matrix, in (0, 1)
    int  S       = 10;     // delay iterations before acceleration kicks in (paper's S≈10)
};

// Called after each recorded step/adjoint iteration of a watchable loop (pd_contact,
// backward_pd_contact, fd_check_contact_stiffness). Returning false aborts the loop early (e.g.
// the viewer window was closed); the caller reads whatever data it needs (tape.positions.back(),
// tape[t], ...) itself rather than having it passed in, since the shape of "current data" differs
// between a forward loop and a backward one.
using StepCallback = std::function<bool(int step, int n_steps)>;

void pd_contact(Object& obj, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix, bool export_obj = true, bool verbose = false, ChebyshevAccel cheby = {}, const StepCallback& on_step = nullptr)
{
    tape.clear();
    tape.record(obj);
    if (export_obj) write_obj_frame(obj, 0, prefix);

    for (int step = 0; step < n_steps; ++step)
    {
        RealVecX x_tilde = obj.x + dt * obj.v;
        const Vec3 dg = (dt * dt) * gravity;
        for (Index i = 0; i < obj.num_particles(); ++i)
            x_tilde.segment<3>(3*i) += dg;

        const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde);
        const Contacts contacts  = detect_contacts(obj, x_tilde, (step + 1) * dt);
        tape.record_contacts(contacts);

        obj.prev_x = obj.x;
        obj.x      = x_tilde;

        RealVecX v_prev = obj.v; // q^(k-1), seeded with q^(0) per the Chebyshev algorithm
        Real     omega  = 1.0;

        for (int k = 0; k < n_iters; ++k)
        {
            const RealVecX b_tilde = construct_velocity_rhs(obj, b_inertia, obj.x, obj.prev_x, dt);

            const RealVecX f = b_tilde - obj.C * obj.v;
            RealVecX g = b_tilde;
            for (const Contact& c : contacts)
            {
                const Vec3 f_i = f.segment<3>(3 * c.particle);
                const Real m_i = obj.mass(3 * c.particle);
                g.segment<3>(3 * c.particle) += update_contact_force(f_i, m_i, collider.velocity, c.normal);
            }

            const RealVecX v_hat = obj.solver->solve(g);
            if (verbose && k==n_iters-1)
            {
                const Real rel_delta = (v_hat - obj.v).norm() / std::max(v_hat.norm(), Real(1e-12));
                if (rel_delta > 1e-4)
                    std::cout << "  step " << step << " iter " << k+1 << " rel_delta = " << rel_delta << "\n";
            }

            if (cheby.enabled)
            {
                Real omega_new;
                if      (k < cheby.S)  omega_new = 1.0;
                else if (k == cheby.S) omega_new = 2.0 / (2.0 - cheby.rho * cheby.rho);
                else                   omega_new = 4.0 / (4.0 - cheby.rho * cheby.rho * omega);
                omega = omega_new;

                const RealVecX v_km1 = v_prev; // q^(k-1)
                v_prev = obj.v;                // this iteration's q^(k) becomes next iter's q^(k-1)
                obj.v  = omega * (v_hat - v_km1) + v_km1;
            }
            else
            {
                obj.v = v_hat;
            }
            obj.x = obj.prev_x + dt * obj.v;
        }

        tape.record(obj);
        if (export_obj && step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);
        if (on_step && !on_step(step, n_steps)) break;
        if (step % 10 == 0) std::cout << "step " << step << "/" << n_steps << "\n";
    }
}

template <typename ParamGrad, typename F>
BackwardGrad<ParamGrad> backward_pd(
    Object&     obj,
    const Tape& tape,
    const Loss& loss,
    int         n_iters_adjoint,
    Real        dt,
    ParamGrad   init_param_grad,
    F&&         accumulate_param_grad)
{
    const int   n_steps = (int)tape.positions.size() - 1;
    const Index dofs    = obj.num_dofs();

    ASSERT((int)loss.dloss_dx.size() == n_steps + 1,
           "loss gradient size " << loss.dloss_dx.size()
           << " != tape size " << tape.positions.size());

    RealVecX  adj        = RealVecX::Zero(dofs);
    ParamGrad param_grad = std::move(init_param_grad);

    for (int t = n_steps; t >= 1; --t)
    {
        const Positions x_plus = Eigen::Map<const Positions>(tape.positions[t].data(), dofs);
        auto [free_grad, z]    = backward_pd_step(obj, x_plus, adj, loss.dloss_dx[t], dt, n_iters_adjoint);
        adj = std::move(free_grad);
        accumulate_param_grad(obj, z, param_grad);

        if (t % 10 == 0) std::cout << "backward step " << t << "/" << n_steps << "\n";
    }

    return { std::move(adj), std::move(param_grad) };
}

struct BackwardGradContact
{
    RealVecX dphi_dv; // dphi/dv0 — gradient of loss w.r.t. initial velocity
    RealVecX dphi_dx; // dphi/dx0 — gradient of loss w.r.t. initial position
    Real     dphi_dk; // dphi/dk  — gradient of loss w.r.t. uniform stiffness
};

BackwardGradContact backward_pd_contact(
    Object&     obj,
    Tape&       tape,
    const Loss& loss,
    const Vec3& gravity,
    int         n_iters_adjoint,
    Real        h,
    const StepCallback& on_step = nullptr)
{
    const int   n_steps = (int)tape.positions.size() - 1;
    const Index dofs    = obj.num_dofs();

    ASSERT((int)loss.dloss_dx.size() == n_steps + 1,
           "loss gradient size " << loss.dloss_dx.size()
           << " != tape size "   << tape.positions.size());
    ASSERT((int)tape.contacts.size() == n_steps,
           "contacts tape size " << tape.contacts.size()
           << " != n_steps "     << n_steps);

    RealVecX dphi_dv = RealVecX::Zero(dofs); // a_t = dL/dv^+_t, seeded from later steps
    RealVecX dphi_dx = RealVecX::Zero(dofs); // b_t = dL/dx^+_t, seeded from later steps
    Real     dphi_dk = 0.0;                  // dphi/dk, accumulated across all steps

    for (int t = n_steps; t >= 1; --t)
    {
        const Positions  x_plus_t   = Eigen::Map<const Positions>(tape.positions[t].data(),       dofs);
        const Velocities v_plus_t   = Eigen::Map<const Velocities>(tape.velocities[t].data(),     dofs);
        const Positions  x_plus_tm1 = Eigen::Map<const Positions>(tape.positions[t - 1].data(),   dofs);
        const Velocities v_plus_tm1 = Eigen::Map<const Velocities>(tape.velocities[t - 1].data(), dofs);
        Contacts& contacts_t        = tape.contacts[t - 1]; 

        const RealVecX b_t = dphi_dx + loss.dloss_dx[t];

        const RealVecX z = compute_adjoint_vector_contact(
            obj, contacts_t, x_plus_t, v_plus_t, x_plus_tm1, v_plus_tm1,
            dphi_dv, h * b_t, gravity, h, n_iters_adjoint);

        dphi_dk += compute_gradient_stiffness_contact(obj, contacts_t, z, x_plus_tm1, v_plus_t, h);

        const ContactSplit split   = split_by_contact(contacts_t, z);
        const RealVecX&     z_perp = split.z_perp; // (I-P) z

        dphi_dv = obj.mass.cwiseProduct(z_perp);

        dphi_dx = h * apply_spring_jacobian(obj, z_perp) - (obj.C * z_perp) / h + b_t;

        for (const Contact& c : contacts_t)
        {
            if (!c.active) continue;
            const Index particle = c.particle;
            const Vec3  vi       = v_plus_t.segment<3>(3 * particle);
            const Vec3  z_i      = z.segment<3>(3 * particle);
            const Vec3  z_perp_i = z_perp.segment<3>(3 * particle);
            const Real  z_dot_n  = c.normal.dot(z_i);
            const Real  m_i      = obj.mass(3 * particle);
            const Vec3  term      = c.d_n * z_perp_i + z_dot_n * m_i * (vi - collider.velocity);
            const Vec3  projected = term - c.axis * c.axis.dot(term); // no-op unless collider is a Cylinder
            dphi_dx.segment<3>(3 * particle) -= c.inv_r * projected;
            // Contact normal is detected at x_tilde = x^- + h*v^- + h^2*g, so it depends on v^-
            // through the same shape operator, scaled by d(x_tilde)/d(v^-) = h*I.
            dphi_dv.segment<3>(3 * particle) -= h * c.inv_r * projected;
        }

        if (on_step && !on_step(t, n_steps)) break;
        if (t % 10 == 0) std::cout << "backward (contact) step " << t << "/" << n_steps << "\n";
    }

    dphi_dx += loss.dloss_dx[0];

    return { std::move(dphi_dv), std::move(dphi_dx), dphi_dk };
}

// ----------------
//    FD CHECK
// ----------------

enum class FDTarget { X0, V0 };

struct FDCheckResult
{
    Real fd;
    Real analytic;
    Real rel_err;
};

FDCheckResult fd_check_contact_direction(
    const std::function<Object()>& build_obj,
    const Tape&     target_tape,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    const RealVecX& direction,
    FDTarget        target,
    const RealVecX& analytic_grad,
    Real            eps = 1e-6)
{
    auto run_loss = [&](Real delta) -> Real
    {
        Object obj = build_obj();
        if (target == FDTarget::X0) obj.x += delta * direction;
        else                        obj.v += delta * direction;

        init_pd_velocity(obj, dt);
        Tape tape;
        pd_contact(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false);
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps);
    const Real loss_minus = run_loss(-eps);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real analytic   = direction.dot(analytic_grad);
    const Real rel_err    = std::abs(fd - analytic)
        / std::max({ std::abs(fd), std::abs(analytic), Real(1e-12) });

    return { fd, analytic, rel_err };
}

RealVecX uniform_axis_direction(Index num_dofs, int axis)
{
    RealVecX d = RealVecX::Zero(num_dofs);
    for (Index i = axis; i < num_dofs; i += 3) d(i) = 1.0;
    return d;
}

void fd_check_contact_offsets(
    const std::function<Object()>& build_obj,
    const Tape&     target_tape,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    Index           num_dofs,
    const RealVecX& dphi_dx0,
    const RealVecX& dphi_dv0,
    Real            eps = 1e-6)
{
    static const char* axis_name[3] = { "x", "y", "z" };
    std::cout << "FD check (uniform offset over all particles, eps=" << eps << "):\n";
    for (int axis = 0; axis < 3; ++axis)
    {
        const RealVecX dir = uniform_axis_direction(num_dofs, axis);

        const FDCheckResult rx = fd_check_contact_direction(
            build_obj, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            sample_every, dir, FDTarget::X0, dphi_dx0, eps);
        const FDCheckResult rv = fd_check_contact_direction(
            build_obj, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            sample_every, dir, FDTarget::V0, dphi_dv0, eps);

        std::cout << "  ." << axis_name[axis]
                   << "  dx0: fd=" << rx.fd << " analytic=" << rx.analytic << " rel_err=" << rx.rel_err
                   << "  |  dv0: fd=" << rv.fd << " analytic=" << rv.analytic << " rel_err=" << rv.rel_err
                   << "\n";
    }
}

FDCheckResult fd_check_contact_stiffness(
    const std::function<Object(Real)>& build_obj_k,
    const Tape&     target_tape,
    Real            k0,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    Real            analytic_dphi_dk,
    Real            eps = 1e-6,
    const StepCallback& on_step = nullptr)
{
    auto run_loss = [&](Real delta) -> Real
    {
        Object obj = build_obj_k(k0 + delta);
        init_pd_velocity(obj, dt);
        Tape tape;
        pd_contact(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false,
                   /*verbose=*/false, /*cheby=*/{}, on_step);
        // on_step can abort pd_contact early (e.g. the viewer window was closed), leaving a
        // truncated tape — Loss asserts equal tape lengths, so bail out with a sentinel instead
        // of crashing; the caller is expected to be exiting right after anyway.
        if ((int)tape.positions.size() != n_steps + 1) return 0.0;
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps);
    const Real loss_minus = run_loss(-eps);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real rel_err    = std::abs(fd - analytic_dphi_dk)
        / std::max({ std::abs(fd), std::abs(analytic_dphi_dk), Real(1e-12) });

    return { fd, analytic_dphi_dk, rel_err };
}

// ----------------
//      MAIN
// ----------------

int main()
{
    ANIM_DIR = ANIM_DIR_DEFAULT;
    clear_folder(ANIM_DIR);

    viewer_open();

    AppConfig cfg; // default member initializers reproduce the old hardcoded literals exactly;
                   // declared outside the loop so edits survive a "Back to Setup" restart

    while (viewer_show_config_screen(cfg))
    {

    bool aborted = false;

    // cloth parameters
    const int width             = cfg.width;
    const int height            = cfg.height;
    const Real stiffness        = cfg.stiffness;
    const Real target_stiffness = cfg.target_stiffness;
    const Vec3 origin           = cfg.origin;
    const Vec3 target_origin    = cfg.target_origin;
    const PinMode pin_mode      = cfg.pin_mode;
    const HangingMode hang_mode = cfg.hang_mode;
    const uint8_t flags         = ClothFlags::STRETCH
                                 | (cfg.flag_shear   ? ClothFlags::SHEAR   : 0)
                                 | (cfg.flag_bending ? ClothFlags::BENDING : 0);
    const Real m_tot            = cfg.m_tot;

    // world parameters
    collider = cfg.collider;

    // physics parameters
    const Vec3 gravity = cfg.gravity;

    // simulation parameters
    const int FPS            = cfg.FPS;
    const int frame_substeps = cfg.frame_substeps;
    const int secs           = cfg.secs;

    // solver parameters
    const int n_iters         = cfg.n_iters;         // forward PD global-local iterations per step
    const int n_iters_adjoint = cfg.n_iters_adjoint; // backward adjoint-vector iterations per step
    const int substeps = FPS * frame_substeps;
    const Real dt      = 1.0 / substeps;
    const int  n_steps = substeps * secs;

    ChebyshevAccel cheby;
    cheby.enabled = false;   // flip to false to fall back to plain Jacobi
    cheby.rho     = 0.9992;
    cheby.S       = 10;

    auto run_target_simulation = [&]() -> Tape
    {
        Object target_obj = cloth(width, height, target_stiffness, target_origin, pin_mode, hang_mode, flags, m_tot);
        init_pd_velocity(target_obj, dt);
        Tape target_tape;
        const StepCallback on_step = [&](int step, int n) -> bool
        {
            if (step % frame_substeps != 0) return true;
            std::ostringstream oss;
            oss << "Target simulation   step " << step << "/" << n;
            viewer_set_scene(target_obj.mesh, nullptr, nullptr,
                              &target_tape.positions.back(), &target_tape.contacts.back(),
                              collider, (step + 1) * dt, oss.str());
            if (!viewer_render_frame()) { aborted = true; return false; }
            return true;
        };
        pd_contact(target_obj, dt, gravity, n_iters, n_steps, frame_substeps, target_tape, "target", false, true, cheby, on_step);
        return target_tape;
    };

    Tape target_tape = run_target_simulation();

    if (aborted) break;

    const bool run_backward = true;
    const bool run_fd_check = false;

    if (run_backward)
    {
        Object guess_obj = cloth(width, height, stiffness, origin, pin_mode, hang_mode, flags, m_tot);
        init_pd_velocity(guess_obj, dt);
        Tape guess_tape;
        const StepCallback guess_on_step = [&](int step, int n) -> bool
        {
            if (step % frame_substeps != 0) return true;
            std::ostringstream oss;
            oss << "Guess simulation   step " << step << "/" << n;
            viewer_set_scene(guess_obj.mesh, &target_tape.positions[step + 1], &target_tape.contacts[step],
                              &guess_tape.positions.back(), &guess_tape.contacts.back(),
                              collider, (step + 1) * dt, oss.str());
            if (!viewer_render_frame()) { aborted = true; return false; }
            return true;
        };
        pd_contact(guess_obj, dt, gravity, n_iters, n_steps, frame_substeps, guess_tape, "guess", false, false, cheby, guess_on_step);

        if (aborted) break;

        Loss loss(guess_tape, target_tape, frame_substeps);
        std::cout << "loss = " << loss.total << "\n";

        const StepCallback backward_on_step = [&](int t, int n) -> bool
        {
            if (t % frame_substeps != 0) return true;
            std::ostringstream oss;
            oss << "Backward pass   step " << t << "/" << n;
            viewer_set_scene(guess_obj.mesh, &target_tape.positions[t], &target_tape.contacts[t - 1],
                              &guess_tape.positions[t], &guess_tape.contacts[t - 1],
                              collider, t * dt, oss.str());
            if (!viewer_render_frame()) { aborted = true; return false; }
            return true;
        };
        const BackwardGradContact grad = backward_pd_contact(guess_obj, guess_tape, loss, gravity, n_iters_adjoint, dt, backward_on_step);

        if (aborted) break;

        const Vec3 dphi_dv0 = Eigen::Map<const PointsX>(
            grad.dphi_dv.data(), guess_obj.num_particles(), 3).colwise().sum().transpose();
        const Vec3 dphi_dx0 = Eigen::Map<const PointsX>(
            grad.dphi_dx.data(), guess_obj.num_particles(), 3).colwise().sum().transpose();

        std::cout << "dphi/dv0 = (" << dphi_dv0.x() << ", " << dphi_dv0.y() << ", " << dphi_dv0.z() << ")\n";
        std::cout << "dphi/dx0 = (" << dphi_dx0.x() << ", " << dphi_dx0.y() << ", " << dphi_dx0.z() << ")\n";
        std::cout << "dphi/dk  = " << grad.dphi_dk << "\n";

        if (run_fd_check)
        {
            // auto build_guess = [&]() -> Object
            // {
            //     return cloth(width, height, stiffness, origin, pin_mode, hang_mode, flags, m_tot);
            // };
            // fd_check_contact_offsets(
            //     build_guess, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            //     frame_substeps, guess_obj.num_dofs(), grad.dphi_dx, grad.dphi_dv);

            auto build_guess_k = [&](Real k) -> Object
            {
                return cloth(width, height, k, origin, pin_mode, hang_mode, flags, m_tot);
            };

            // FD reruns stay headless (no new geometry drawn) — this just pumps the window and
            // updates the status line so it doesn't look frozen while the reruns compute.
            const StepCallback fd_heartbeat = [&](int step, int n) -> bool
            {
                if (step % frame_substeps != 0) return true;
                std::ostringstream oss;
                oss << "FD check (stiffness)   step " << step << "/" << n;
                viewer_set_status(oss.str());
                if (!viewer_render_frame()) { aborted = true; return false; }
                return true;
            };

            const std::vector<Real> epss = { 1e-8, 1e-9, 1e-10 };// { 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9 };
            std::vector<FDCheckResult> results;
            for (const Real eps : epss)
            {
                std::cout << "running fd check with eps=" << eps << "\n";
                results.push_back(fd_check_contact_stiffness(
                    build_guess_k, target_tape, stiffness, dt, gravity, n_iters, n_steps, frame_substeps,
                    frame_substeps, grad.dphi_dk, eps, fd_heartbeat));
                if (aborted) break;
            }

            if (aborted) break;

            for (size_t i = 0; i < epss.size(); ++i)
            {
                const FDCheckResult& rk = results[i];
                std::cout << "  k   dk: eps=" << epss[i] << " fd=" << rk.fd
                          << " analytic=" << rk.analytic << " rel_err=" << rk.rel_err << "\n";
            }
        }

        if (!viewer_interactive_playback(guess_obj.mesh, target_tape, guess_tape, collider, dt, 1, FPS * frame_substeps))
            break; // window closed; "Back to Setup" falls through and loops back to the config screen
    }

    } // while (viewer_show_config_screen(cfg))

    viewer_close();
    return 0;
}