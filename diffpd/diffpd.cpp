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

    static Constraint makeSpring2(Real k, Real l, ParticleId i1, ParticleId i2) 
    {
        Constraint c;
        c.type = SpringType::Spring2;
        c.k = k;
        c.l = l;
        c.spring2.i1 = i1;
        c.spring2.i2 = i2;
        return c;
    }

    static Constraint makeSpring1(Real k, Real l, ParticleId i, const Vec3& xbar) 
    {
        Constraint c;
        c.type = SpringType::Spring1;
        c.k = k;
        c.l = l;
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

    SparseMat L;       // L = M/h^2 + sum_i k_i G_i^T G_i   (SPD, constant)
    Cholesky  solver;  // factor of L;

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
    Index   width, Index   height,
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

    PointsX X(N, 3);
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
            X.row(grid(i, j)) = (origin + Vec3(i * sx, 0.0, j * sz)).transpose();

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
            obj.x.segment<3>(3 * dof[v]) = X.row(v).transpose();

    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    // --- mass: m_tot distributed uniformly over n free particles ----------
    obj.mass = MassDiag::Constant(3 * n, m_tot / Real(n));

    // ---- bake export mesh (after dof[] and X are built) ------------------
    Mesh& mesh = obj.mesh;
    std::vector<Index> mesh_vid(N, -1);   // grid vertex -> mesh vertex index

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
                mv.dof = -Index(mesh.pinned_rest.size()) - 1;
                mesh.pinned_rest.push_back(X.row(v).transpose());
            }
            mesh_vid[v] = Index(mesh.vertices.size());
            mesh.vertices.push_back(mv);
        }

    // structural edges only (both endpoints, pinned or not)
    auto mesh_edge = [&](Index a, Index b) 
    {
        mesh.edges.emplace_back(mesh_vid[a], mesh_vid[b]);
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
        const Real l  = (X.row(a) - X.row(b)).norm();
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
            const Vec3   xbar   = X.row(anchor).transpose();
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
        const Real diag = std::sqrt(sx * sx + sz * sz);
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