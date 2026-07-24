#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cstdint>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <memory>

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

using Positions  = RealVecX;
using Velocities = RealVecX;
using RestMesh   = PointsX;

// ----------------
//      MESH
// ----------------

struct Object; // SimMesh::position() needs Object; defined below, method body after Object.

// Named SimMesh (not Mesh) to avoid colliding with raylib's own global `struct Mesh`.
struct SimMesh
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

using Constraints = std::vector<Constraint>;

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

    SimMesh mesh;

    Object() = default;

    Object(const Object&)            = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&)                 = default;
    Object& operator=(Object&&)      = default;

    Index num_particles() const { return x.rows() / 3; }
    Index num_dofs()      const { return x.rows();     }
};

inline Vec3 SimMesh::position(const Object& obj, Index vi) const
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
//    COLLIDER
// ----------------

enum class ColliderType { Sphere, Cylinder, Plane, Capsule };

struct Collider
{
    ColliderType type = ColliderType::Sphere;

    // Sphere: center + radius
    Vec3 sphere_center = Vec3(0.0, -10.0, 0.0);
    Real sphere_radius = 1.0;

    // Infinite cylinder: point on the axis + unit axis direction + radius
    Vec3 cylinder_origin = Vec3::Zero();
    Vec3 cylinder_axis   = Vec3::UnitY();
    Real cylinder_radius = 1.0;

    // Plane: point on the plane + unit outward normal
    Vec3 plane_origin = Vec3::Zero();
    Vec3 plane_normal = Vec3::UnitY();

    // Capsule: segment endpoints + radius (cylindrical body capped by two hemispheres)
    Vec3 capsule_p0     = Vec3::Zero();
    Vec3 capsule_p1     = Vec3::UnitY();
    Real capsule_radius = 1.0;

    Vec3 velocity = Vec3::Zero(); // shared constant translation velocity (m/s), whichever shape is active
};

inline Collider make_sphere(Vec3 center, Real radius, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type          = ColliderType::Sphere;
    c.sphere_center = center;
    c.sphere_radius = radius;
    c.velocity      = velocity;
    return c;
}

inline Collider make_cylinder(Vec3 origin, Vec3 axis, Real radius, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type            = ColliderType::Cylinder;
    c.cylinder_origin = origin;
    c.cylinder_axis   = axis;
    c.cylinder_radius = radius;
    c.velocity        = velocity;
    return c;
}

inline Collider make_plane(Vec3 origin, Vec3 normal, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type         = ColliderType::Plane;
    c.plane_origin = origin;
    c.plane_normal = normal;
    c.velocity     = velocity;
    return c;
}

inline Collider make_capsule(Vec3 p0, Vec3 p1, Real radius, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type           = ColliderType::Capsule;
    c.capsule_p0     = p0;
    c.capsule_p1     = p1;
    c.capsule_radius = radius;
    c.velocity       = velocity;
    return c;
}

// ----------------
//  CLOTH CONFIG
// ----------------

// How an object is anchored.
//   For cloth: NONE = free fall, CORNERS = two top corners, ROW = entire first row.
enum class PinMode { NONE, CORNERS, ROW };

// How the cloth grid is initially oriented.
//   HORIZONTAL: laid flat in the XZ plane (current/default behavior) — falls and swings under gravity.
//   VERTICAL:   laid in the XY plane, j=0 row at the top, hanging straight down (-Y) from there.
enum class HangingMode { HORIZONTAL, VERTICAL };

namespace ClothFlags
{
    constexpr uint8_t STRETCH = 1 << 0;
    constexpr uint8_t SHEAR   = 1 << 1;
    constexpr uint8_t BENDING = 1 << 2;
    constexpr uint8_t ALL     = STRETCH | SHEAR | BENDING;
}

// Forward declaration only — the body (grid generation) lives in diffpd.cpp. Declared here (with
// the full default-argument list) so diffpd_viewer.cpp can call it to rebuild a live preview each
// frame without ever including diffpd.cpp (keeping diffpd_viewer.cpp the only raylib-aware file).
Object cloth(
    Index       width,
    Index       height,
    Real        stiffness    = DEFAULT_STIFFNESS,
    Vec3        origin       = Vec3::Zero(),
    PinMode     pin_mode     = PinMode::CORNERS,
    HangingMode hanging_mode = HangingMode::HORIZONTAL,
    uint8_t     flags        = ClothFlags::ALL,
    Real        m_tot        = 1.0);

// ----------------
//   APP CONFIG
// ----------------

// Pre-fills every shape's params (the literals main() used to leave commented out for the three
// inactive shapes, plus whichever was actually active) so switching collider shape in the config
// screen never lands on Collider's own generic defaults.
inline Collider default_config_collider()
{
    Collider c;
    c.type            = ColliderType::Plane;
    c.sphere_center   = Vec3(1.0, -1.0, 1.0);
    c.sphere_radius   = 0.5;
    c.cylinder_origin = Vec3(5.0, -6.0, 5.0);
    c.cylinder_axis   = Vec3::UnitX();
    c.cylinder_radius = 3.0;
    c.plane_origin    = Vec3(0.0, -1.4, 0.0);
    c.plane_normal    = Vec3::UnitY();
    c.capsule_p0      = Vec3(5.0, -10.0, 0.0);
    c.capsule_p1      = Vec3(5.0, -5.0, 0.0);
    c.capsule_radius  = 3.0;
    c.velocity        = Vec3::Zero();
    return c;
}

// Every knob main() used to hardcode, now editable live in the pre-run config screen. Default
// member initializers reproduce the old literals exactly, so accepting all defaults and hitting
// Run reproduces the previous hardcoded behavior byte-for-byte. Chebyshev acceleration is
// intentionally not here (out of scope) — main() keeps hardcoding it.
struct AppConfig
{
    // cloth
    int         width            = 30;
    int         height           = 30;
    Real        stiffness        = 1.0;
    Real        target_stiffness = 2.0;
    Vec3        origin           = Vec3(0.0, 0.0, 0.0);
    Vec3        target_origin    = Vec3(0.0, 0.0, 0.0);
    PinMode     pin_mode         = PinMode::ROW;
    HangingMode hang_mode        = HangingMode::HORIZONTAL;
    bool        flag_shear       = true;  // ClothFlags::SHEAR   (STRETCH always on, not stored)
    bool        flag_bending     = true;  // ClothFlags::BENDING
    Real        m_tot            = 0.1;

    // collider (all four shapes' params live here simultaneously, exactly like Collider itself)
    Collider collider = default_config_collider();

    // physics
    Vec3 gravity = Vec3::UnitY() * -9.81;

    // simulation / solver
    int FPS             = 24;
    int frame_substeps  = 4;
    int secs            = 5;
    int n_iters         = 100; // forward PD global-local iterations per step
    int n_iters_adjoint = 100; // backward adjoint-vector iterations per step
};

// ----------------
//    CONTACT
// ----------------

struct Contact
{
    ParticleId particle;
    Vec3       normal; // unit outward contact normal
    Real       inv_r;  // curvature scale (1/dist to the collider's center/axis); 0 for a plane
    bool       active; // set in the backward pass: true if the contact is pressing (d_n < 0)
    Real       d_n;    // set in the backward pass: f_i . normal, cached for the d(normal)/dx term
    Vec3       axis = Vec3::Zero(); // unit collider axis; zero unless the collider is a Cylinder
};

using Contacts = std::vector<Contact>;

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
