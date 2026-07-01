#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

#include <cstdint>
#include <vector>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>
#include <functional>

namespace fs = std::filesystem;

#define WARNING(message) \
    do { \
        std::ostringstream _oss; \
        _oss << message; \
        std::fprintf(stderr, "[WARNING] %s:%d: %s\n", \
                     __FILE__, __LINE__, _oss.str().c_str()); \
    } while (0)

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::ostringstream _oss; \
            _oss << message; \
            std::fprintf(stderr, "[ASSERT] %s:%d: (%s) %s\n", \
                         __FILE__, __LINE__, #condition, _oss.str().c_str()); \
            std::abort(); \
        } \
    } while (0)

// ----------------
//      TYPES
// ----------------

using Real         = double;
using Index        = Eigen::Index;
using VertexId     = int;
using ParticleId   = int;
using ConstraintId = int;

using Vec3 = Eigen::Matrix<Real, 3, 1>;

using Mat3 = Eigen::Matrix<Real, 3, 3>;

using RealVecX = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

using PointsX = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;

using SparseMat = Eigen::SparseMatrix<Real>;
using Triplet   = Eigen::Triplet<Real>;

using Cholesky = Eigen::SimplicialLLT<SparseMat>;

using MassDiag = RealVecX;

using Edge  = std::pair<VertexId, VertexId>;
using Edges = std::vector<Edge>;

constexpr Real DEFAULT_STIFFNESS = 1e3;

struct Constraint;
using Constraints = std::vector<Constraint>;
struct Object;
struct SimulationState;

using Positions  = RealVecX;
using Velocities = RealVecX;
using RestMesh   = PointsX;

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
//      MESH
// ----------------

struct Mesh 
{
    struct Vertex 
    {
        ParticleId dof;
    };

    std::vector<Vertex> vertices;              // output order == index order
    std::vector<Vec3>   pinned_rest;           // fixed positions for pinned verts
    std::vector<std::pair<Index,Index>> edges; // indices into `vertices`

    Vec3 position(const Object& obj, Index vi) const; // defined after Object
};

// ----------------
//   CONSTRAINTS
// ----------------

enum class SpringType : std::uint8_t 
{
    Spring2,   // both endpoints are free particles
    Spring1,   // one free particle anchored to a pinned vertex
};

struct Constraint
{
    SpringType type;
    Real       k; // stiffness
    Real       l; // rest length
    Mat3       gamma;  // local Jacobian factor: (k*l/||e||) * P_perp
    Vec3       p_star; // normalized spring correction: l * (e / ||e||)
    Vec3       e;      // edge vector

    union
    {
        struct
        {
            ParticleId i1;
            ParticleId i2;
        } spring2;

        struct
        {
            ParticleId i;
            Real       xbar[3];
        } spring1;
    };

    static Constraint make(SpringType type, Real k, Real l)
    {
        Constraint c;
        c.type   = type;
        c.k      = k;
        c.l      = l;
        c.gamma  = Mat3::Zero();
        c.p_star = Vec3::Zero();
        c.e      = Vec3::Zero();
        return c;
    }

    static Constraint makeSpring2(Real k, Real l, ParticleId i1, ParticleId i2)
    {
        Constraint c = make(SpringType::Spring2, k, l);
        c.spring2.i1 = i1;
        c.spring2.i2 = i2;
        return c;
    }

    static Constraint makeSpring1(Real k, Real l, ParticleId i, const Vec3& xbar)
    {
        Constraint c = make(SpringType::Spring1, k, l);
        c.spring1.i = i;
        c.spring1.xbar[0] = xbar.x();
        c.spring1.xbar[1] = xbar.y();
        c.spring1.xbar[2] = xbar.z();
        return c;
    }
};

struct Object 
{
    Positions  x;        // current positions       (3n)
    Velocities v;        // current velocities      (3n)
    Positions  prev_x;   // x at start of step

    MassDiag    mass;    // diagonal of M,          (3n) 
    Constraints constraints;

    SparseMat L;                       // L = M/h^2 + sum_i k_i G_i^T G_i   (SPD, constant)
    SparseMat C;                       // h^2 * K — elastic block; only set by construct_velocity_lhs
    std::unique_ptr<Cholesky> solver;  // factor of L; heap-allocated so Object stays moveable

    Mesh mesh; 

    Object() = default;

    Object(const Object&)            = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&)                 = default;
    Object& operator=(Object&&)      = default;

    Index num_particles() const { return x.rows() / 3; }
    Index num_dofs()      const { return x.rows();     }
};

inline Vec3 Mesh::position(const Object& obj, Index vi) const
{
    const Vertex& vert = vertices[vi];
    if (vert.dof >= 0)
    {
        ASSERT(3 * vert.dof + 2 < obj.x.rows(), "mesh dof out of range for obj.x");
        return obj.x.segment<3>(3 * vert.dof);
    }
    return pinned_rest[ -(vert.dof + 1) ];
}

// ----------------
//    CONTACT
// ----------------

struct Contact
{
    ParticleId particle;
    Vec3       normal; // unit outward contact normal
    bool       active; // set in the backward pass: true if the contact is pressing (d_n < 0)
};

using Contacts = std::vector<Contact>;

Contacts detect_contacts(const Object& obj)
{
    Contacts contacts;
    const Vec3 ground_point  = Vec3(0.0, -7.0, 0.0);
    const Vec3 ground_normal = Vec3(0.0, 1.0, 0.2).normalized();

    for (Index i = 0; i < obj.num_particles(); ++i)
    {
        const Vec3 pos  = obj.x.segment<3>(3*i);
        const Real dist = (pos - ground_point).dot(ground_normal);
        if (dist < 0.0)
        {
            contacts.push_back({static_cast<ParticleId>(i), ground_normal, false});
        }
    }

    return contacts;
}

Vec3 update_contact_force(const Vec3& d, const Vec3& n)
{
    const Real d_n = d.dot(n);
    if (d_n >= 0.0) return Vec3::Zero();
    return -d_n * n;
}

// ----------------
//       TAPE
// ----------------

struct Tape
{
    std::vector<PointsX> positions;   // positions: one Nx3 matrix per timestep
    std::vector<PointsX> velocities;  // velocities: one Nx3 matrix per timestep
    std::vector<Contacts> contacts;   // contacts[t] = contacts active during step t -> t+1 (size == n_steps)

    void clear()
    {
        positions.clear();
        velocities.clear();
        contacts.clear();
    }

    void record(const Object& obj)
    {
        const Index n = obj.num_particles();
        PointsX pos = Eigen::Map<const PointsX>(obj.x.data(), n, 3);
        PointsX vel = Eigen::Map<const PointsX>(obj.v.data(), n, 3);
        positions.push_back(pos);
        velocities.push_back(vel);
    }

    void record_contacts(const Contacts& c) { contacts.push_back(c); }
};

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

// How an object is anchored. 
//   For cloth: NONE = free fall, CORNERS = two top corners, ROW = entire first row.
enum class PinMode { NONE, CORNERS, ROW };

namespace ClothFlags 
{
    constexpr uint8_t STRETCH = 1 << 0;
    constexpr uint8_t SHEAR   = 1 << 1;
    constexpr uint8_t BENDING = 1 << 2;
    constexpr uint8_t ALL     = STRETCH | SHEAR | BENDING;
}

constexpr Real cloth_size = 10.0;

Object cloth(
    Index   width, 
    Index   height,
    Real    stiffness = DEFAULT_STIFFNESS,
    Vec3    origin    = Vec3::Zero(),
    PinMode pin_mode  = PinMode::CORNERS,
    uint8_t flags     = ClothFlags::ALL,
    Real    m_tot     = 1.0)
{
    ASSERT(flags & ClothFlags::STRETCH, "cloth must have stretch constraints enabled");

    const Index N  = width * height;
    const Real  sx = cloth_size / Real(width  - 1);
    const Real  sz = cloth_size / Real(height - 1);

    auto grid = [height](Index i, Index j) { return i * height + j; };

    std::vector<Vec3> pos(N);
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
            pos[grid(i, j)] = origin + Vec3(i * sx, 0.0, j * sz);

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
    Mesh& mesh = obj.mesh;

    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
        {
            const Index v = grid(i, j);
            Mesh::Vertex mv;
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
    const Mesh& mesh = obj.mesh;

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

RealVecX construct_rhs(const Object& obj, const RealVecX& x, const RealVecX& b_inertia)
{
    RealVecX b = b_inertia;

    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 e   = x.segment<3>(3*i1) - x.segment<3>(3*i2);
            const Vec3 p   = c.l * e / e.norm();
            b.segment<3>(3*i1) += c.k * p;
            b.segment<3>(3*i2) -= c.k * p;
        }
        else // SpringType::Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;
            const Vec3 e  = x.segment<3>(3*i) - xbar;
            const Vec3 p  = c.l * e / e.norm();
            b.segment<3>(3*i) += c.k * (xbar + p);
        }
    }

    return b;
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
    const Real h2            = h * h;
    const RealVecX b_elastic = construct_rhs(obj, x_k, RealVecX::Zero(obj.num_dofs()));
    const RealVecX b         = b_inertia + h2 * b_elastic;
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

RealVecX construct_backward_rhs(
    const Object&   obj,
    const RealVecX& z,
    const RealVecX& dloss_dx,
    const RealVecX& dloss_dx_t)
{
    RealVecX b = RealVecX::Zero(z.rows());

    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 d   = c.gamma * (z.segment<3>(3*i1) - z.segment<3>(3*i2));
            b.segment<3>(3*i1) += d;
            b.segment<3>(3*i2) -= d;
        }
        else // SpringType::Spring1
        {
            const Index i = c.spring1.i;
            b.segment<3>(3*i) += c.gamma * z.segment<3>(3*i);
        }
    }

    return b + dloss_dx + dloss_dx_t;
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
    const Vec3 dg = h * h * gravity;
    for (Index i = 0; i < obj.num_particles(); ++i)
        x_tilde.segment<3>(3*i) += dg;

    const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde);
    const RealVecX b_tilde   = construct_velocity_rhs(obj, b_inertia, x_plus, x_minus, h);
    const RealVecX f         = b_tilde - obj.C * v_plus;

    for (Contact& c : contacts)
    {
        const Vec3 f_i = f.segment<3>(3 * c.particle);
        const Real d_n = f_i.dot(c.normal);
        c.active = (d_n < 0.0);
    }
}

RealVecX construct_backward_contact_rhs(
    const Object&   obj,
    const Contacts& contacts,
    const RealVecX& z,
    const RealVecX& dloss_dv,
    const RealVecX& dloss_dv_t,
    Real            h)
{
    // Applies the off-diagonal (ΔA + ΔR) block of the adjoint system, transposed.
    //
    // With frictionless normal contact the step Hessian is non-symmetric:
    //     ∇²g = L - (I-P) h²ΔA - P C ,   P = block-diag(n nᵀ) over active contacts,
    //     C  = h²K (obj.C),   h²ΔA = Σ_i Gᵢᵀ Γᵢ Gᵢ  (the spring local Jacobian).
    // At an active contact node the *normal* direction is governed by the contact
    // response (P C) and no longer by elasticity ((I-P) removes it) — the two are not
    // additive. The adjoint solves (∇²g)ᵀ z = seed, so we apply the transpose:
    //     (L - ∇²g)ᵀ z = h²ΔA ((I-P) z) + C (P z).
    const Real  h2   = h * h;
    const Index dofs = z.rows();

    // Split z at active contacts:  z = (I-P) z + P z.
    RealVecX z_perp = z;                        // (I-P) z : normal component removed at contacts
    RealVecX Pz     = RealVecX::Zero(dofs);     // P z     : only the normal component at contacts
    for (const Contact& c : contacts)
    {
        if (!c.active) continue;
        const Vec3 zi = z.segment<3>(3 * c.particle);
        const Vec3 pn = c.normal * c.normal.dot(zi);
        Pz.segment<3>(3 * c.particle)     = pn;
        z_perp.segment<3>(3 * c.particle) = zi - pn;
    }

    // Spring term: h²ΔA · (I-P) z   (elastic local Jacobian, applied to z_perp)
    RealVecX b = RealVecX::Zero(dofs);
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 d = h2 * (c.gamma * (z_perp.segment<3>(3*i1) - z_perp.segment<3>(3*i2)));
            b.segment<3>(3*i1) += d;
            b.segment<3>(3*i2) -= d;
        }
        else // SpringType::Spring1
        {
            const Index i = c.spring1.i;
            b.segment<3>(3*i) += h2 * (c.gamma * z_perp.segment<3>(3*i));
        }
    }

    // Contact term: C · (P z)   (global elastic product; couples contact nodes to neighbours)
    if (!contacts.empty())
        b += obj.C * Pz;

    return b + dloss_dv + dloss_dv_t;
}

RealVecX compute_adjoint_vector(
    Object&          obj,
    const Positions& x_plus,
    const RealVecX&  dloss_dx,
    const RealVecX&  dloss_dx_t,
    int              n_iters)
{
    precompute_constraints_local_derivative(obj, x_plus);

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    for (int k = 0; k < n_iters; ++k)
    {
        const RealVecX b_back = construct_backward_rhs(obj, z, dloss_dx, dloss_dx_t);
        z = obj.solver->solve(b_back);
    }
    return z;
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
    int               n_iters)
{
    precompute_constraints_local_derivative(obj, x_plus);
    precompute_contacts_local_derivative(contacts, obj, x_plus, v_plus, x_minus, v_minus, gravity, h);

    constexpr Real kConvergenceTol = 1e-8;

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    Real rel_residual = 0.0;
    for (int k = 0; k < n_iters; ++k)
    {
        const RealVecX b_back = construct_backward_contact_rhs(obj, contacts, z, dloss_dv, dloss_dv_t, h);
        const RealVecX z_new  = obj.solver->solve(b_back);
        rel_residual = (z_new - z).norm() / std::max(z_new.norm(), Real(1e-12));
        z = z_new;
    }
    if (n_iters > 0 && rel_residual > kConvergenceTol)
        WARNING("compute_adjoint_vector_contact: adjoint iteration did not converge after "
                << n_iters << " iters (relative residual = " << rel_residual << ")");

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
    int              n_iters)
{
    RealVecX z = compute_adjoint_vector(obj, x_plus, dloss_dx, dloss_dx_t, n_iters);
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

    for (int k = 0; k < n_iters; ++k)
    {
        const RealVecX b = construct_rhs(obj, obj.x, b_inertia);
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

void pd_contact(Object& obj, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix, bool export_obj = true)
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
        const Contacts contacts  = detect_contacts(obj);
        tape.record_contacts(contacts);

        obj.prev_x = obj.x;
        obj.x      = x_tilde;

        for (int k = 0; k < n_iters; ++k)
        {
            const RealVecX b_tilde = construct_velocity_rhs(obj, b_inertia, obj.x, obj.prev_x, dt);

            const RealVecX f = b_tilde - obj.C * obj.v;
            RealVecX g = b_tilde;
            for (const Contact& c : contacts)
            {
                const Vec3 f_i = f.segment<3>(3 * c.particle);
                g.segment<3>(3 * c.particle) += update_contact_force(f_i, c.normal);
            }

            obj.v = obj.solver->solve(g);
            obj.x = obj.prev_x + dt * obj.v;
        }

        tape.record(obj);
        if (export_obj && step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);
        if (step % 10 == 0) std::cout << "step " << step << "/" << n_steps << "\n";
    }
}

template <typename ParamGrad, typename F>
BackwardGrad<ParamGrad> backward_pd(
    Object&     obj,
    const Tape& tape,
    const Loss& loss,
    int         n_iters,
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
        auto [free_grad, z]    = backward_pd_step(obj, x_plus, adj, loss.dloss_dx[t], dt, n_iters);
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
};

BackwardGradContact backward_pd_contact(
    Object&     obj,
    Tape&       tape,
    const Loss& loss,
    const Vec3& gravity,
    int         n_iters,
    Real        h)
{
    const int   n_steps = (int)tape.positions.size() - 1;
    const Index dofs    = obj.num_dofs();

    ASSERT((int)loss.dloss_dx.size() == n_steps + 1,
           "loss gradient size " << loss.dloss_dx.size()
           << " != tape size " << tape.positions.size());
    ASSERT((int)tape.contacts.size() == n_steps,
           "contacts tape size " << tape.contacts.size()
           << " != n_steps " << n_steps);

    RealVecX dphi_dv = RealVecX::Zero(dofs); // dphi/dv^+_t, seeded from later steps
    RealVecX dphi_dx = RealVecX::Zero(dofs); // dphi/dx^+_t, seeded from later steps

    for (int t = n_steps; t >= 1; --t)
    {
        const Positions  x_plus_t   = Eigen::Map<const Positions>(tape.positions[t].data(),       dofs);
        const Velocities v_plus_t   = Eigen::Map<const Velocities>(tape.velocities[t].data(),     dofs);
        const Positions  x_plus_tm1 = Eigen::Map<const Positions>(tape.positions[t - 1].data(),   dofs);
        const Velocities v_plus_tm1 = Eigen::Map<const Velocities>(tape.velocities[t - 1].data(), dofs);
        const RealVecX dloss_dv_t   = h * loss.dloss_dx[t]; // per-step position loss -> velocity space
        Contacts& contacts_t        = tape.contacts[t - 1]; // contacts active during step t-1 -> t (gamma written in place)

        const RealVecX z = compute_adjoint_vector_contact(
            obj, contacts_t, x_plus_t, v_plus_t, x_plus_tm1, v_plus_tm1,
            dphi_dv, dloss_dv_t, gravity, h, n_iters);

        dphi_dv = obj.mass.cwiseProduct(z);
        dphi_dx = dphi_dv / h;

        if (t % 10 == 0) std::cout << "backward (contact) step " << t << "/" << n_steps << "\n";
    }

    // loss.dloss_dx[0] is x0's *direct* loss gradient (the t=0 frame, if sampled) -- the
    // t=n_steps..1 loop above only propagates loss contributions from frames t>=1 back
    // through the dynamics, so this term has to be added separately.
    dphi_dx += loss.dloss_dx[0];

    return { std::move(dphi_dv), std::move(dphi_dx) };
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

// Central-difference check of the loss's directional derivative along `direction` (a
// perturbation applied to x0 or v0), against the analytic gradient dotted with that same
// direction. A one-hot `direction` gives a single-DOF spot check; a per-axis indicator
// (1 at every particle's x, or y, or z) gives the aggregate/uniform-offset gradient — i.e.
// the same quantity as `dphi_dx0`/`dphi_dv0`'s colwise().sum() in main().
// build_obj() must return a fresh, unperturbed Object built with the same cloth params
// used to produce `analytic_grad`.
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
    const Real fd       = (loss_plus - loss_minus) / (2.0 * eps);
    const Real analytic = direction.dot(analytic_grad);
    const Real rel_err  = std::abs(fd - analytic)
        / std::max({ std::abs(fd), std::abs(analytic), Real(1e-12) });

    return { fd, analytic, rel_err };
}

// direction(i) = 1 at every DOF whose local component is `axis` (0=x, 1=y, 2=z), 0 elsewhere —
// i.e. a uniform per-particle offset along one global axis.
RealVecX uniform_axis_direction(Index num_dofs, int axis)
{
    RealVecX d = RealVecX::Zero(num_dofs);
    for (Index i = axis; i < num_dofs; i += 3) d(i) = 1.0;
    return d;
}

// Runs fd_check_contact_direction with a uniform per-axis offset (matching how main()
// aggregates grad.dphi_dx/dphi_dv into dphi_dx0/dphi_dv0 via colwise().sum()), for all
// three axes against both dphi/dx0 and dphi/dv0, and prints a comparison table.
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

// ----------------
//      MAIN
// ----------------

int main()
{
    ANIM_DIR = ANIM_DIR_DEFAULT;
    clear_folder(ANIM_DIR);

    // cloth parameters
    const int width             = 10;
    const int height            = 10;
    const Real stiffness        = 1.0;
    const Real target_stiffness = 1.0;
    const Vec3 origin           = Vec3(0.4,  0.0, 0.2);
    const Vec3 target_origin    = Vec3(0.0,  0.0, 0.0);
    const PinMode pin_mode      = PinMode::NONE;
    const uint8_t flags         = ClothFlags::ALL;
    const Real m_tot            = 1.0;

    // physics parameters
    const Vec3 gravity = Vec3::UnitY() * -9.81;

    // simulation parameters
    const int FPS            = 24;
    const int frame_substeps = 3;
    const int secs           = 2;

    // solver parameters
    const int n_iters  = 20;
    const int substeps = FPS * frame_substeps;
    const Real dt      = 1.0 / substeps;
    const int  n_steps = substeps * secs;

    auto run_target_simulation = [&]() -> Tape
    {
        Object target_obj = cloth(width, height, target_stiffness, target_origin, pin_mode, flags, m_tot);
        init_pd_velocity(target_obj, dt);
        Tape target_tape;
        pd_contact(target_obj, dt, gravity, n_iters, n_steps, frame_substeps, target_tape, "target");
        return target_tape;
    };

    Tape target_tape = run_target_simulation();

    Object guess_obj = cloth(width, height, stiffness, origin, pin_mode, flags, m_tot);
    init_pd_velocity(guess_obj, dt);
    Tape guess_tape;
    pd_contact(guess_obj, dt, gravity, n_iters, n_steps, frame_substeps, guess_tape, "guess");

    Loss loss(guess_tape, target_tape, frame_substeps);
    std::cout << "loss = " << loss.total << "\n";

    const BackwardGradContact grad = backward_pd_contact(guess_obj, guess_tape, loss, gravity, n_iters, dt);

    const Vec3 dphi_dv0 = Eigen::Map<const PointsX>(
        grad.dphi_dv.data(), guess_obj.num_particles(), 3).colwise().sum().transpose();
    const Vec3 dphi_dx0 = Eigen::Map<const PointsX>(
        grad.dphi_dx.data(), guess_obj.num_particles(), 3).colwise().sum().transpose();

    std::cout << "dphi/dv0 = (" << dphi_dv0.x() << ", " << dphi_dv0.y() << ", " << dphi_dv0.z() << ")\n";
    std::cout << "dphi/dx0 = (" << dphi_dx0.x() << ", " << dphi_dx0.y() << ", " << dphi_dx0.z() << ")\n";

    const bool run_fd_check = true;
    if (run_fd_check)
    {
        auto build_guess = [&]() -> Object
        {
            return cloth(width, height, stiffness, origin, pin_mode, flags, m_tot);
        };
        fd_check_contact_offsets(
            build_guess, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            frame_substeps, guess_obj.num_dofs(), grad.dphi_dx, grad.dphi_dv);
    }

    // auto run_guess_simulation = [&]() -> Tape
    // {
    //     Object guess_obj = cloth(width, height, stiffness, origin, pin_mode, flags, m_tot);
    //     init_pd_velocity(guess_obj, dt);
    //     Tape guess_tape;
    //     pd_contact(guess_obj, dt, gravity, n_iters, n_steps, frame_substeps, guess_tape, "guess");
    //     return guess_tape;
    // };

    // Tape target_tape = run_target_simulation();

    // Object obj = cloth(width, height, stiffness, origin, pin_mode, flags, m_tot);
    // init_pd(obj, dt);
    // Tape tape;
    // pd(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, "guess");

    // Loss loss(tape, target_tape, frame_substeps);
    // std::cout << "loss = " << loss.total << "\n";

    // auto grad_stiffness = [](const Object& obj, const RealVecX& z, Real& acc) 
    // {
    //     acc += compute_gradient_stiffness(obj, z);
    // };

    // const BackwardGrad<Real> grad_k = backward_pd(obj, tape, loss, n_iters, dt,
    //     Real(0), grad_stiffness);

    // std::cout << "stiffness gradient = " << grad_k.param_grad << "\n";

    // auto grad_anchor_positions = [](const Object& obj, const RealVecX& z, Vec3& acc) 
    // {
    //     acc += compute_gradient_pinned_vertices(obj, z);
    // };

    // const BackwardGrad<Vec3> grad = backward_pd(obj, tape, loss, n_iters, dt,
    //     Vec3(Vec3::Zero()), grad_anchor_positions);

    // const Vec3 free_offset_grad = Eigen::Map<const PointsX>(
    //     grad.free_grad.data(), obj.num_particles(), 3).colwise().sum().transpose();
    // const Vec3 total_offset_grad = free_offset_grad + grad.param_grad;

    // std::cout << "free_offset_grad   = (" << free_offset_grad.x()  << ", " << free_offset_grad.y()  << ", " << free_offset_grad.z()  << ")\n";
    // std::cout << "anchor_offset_grad = (" << grad.param_grad.x()   << ", " << grad.param_grad.y()   << ", " << grad.param_grad.z()   << ")\n";
    // std::cout << "total_offset_grad  = (" << total_offset_grad.x() << ", " << total_offset_grad.y() << ", " << total_offset_grad.z() << ")\n";

    // const auto t0 = std::chrono::steady_clock::now();
    // pd(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape);
    // const auto t1 = std::chrono::steady_clock::now();

    // const Real wall_s = std::chrono::duration<Real>(t1 - t0).count();
    // const Real sim_s  = dt * n_steps;
    // std::cout << "wall time: " << wall_s << " s\n"
    //           << "sim time:  " << sim_s  << " s\n"
    //           << "ratio:     " << sim_s / wall_s << "x"  "\n";

    return 0;
}