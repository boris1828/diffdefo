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
#include <functional>

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

    // Grid topology: vertices[] is laid out row-major as i*height+j, i in [0,width), j in
    // [0,height) — set by cloth()/skirt() alongside vertices/edges. Lets renderers reconstruct quad
    // adjacency (e.g. for a subdivided smooth-shaded surface) without re-deriving it from edges[].
    Index width  = 0;
    Index height = 0;

    // True when the i axis wraps around (i=width-1 is adjacent to i=0, forming a closed ring) —
    // set by skirt(), left false by cloth(). Grid-based renderers need this to close the seam
    // between the last and first column instead of leaving it open like an ordinary sheet.
    bool wrap_i = false;

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

// None is appended last (rather than first) so existing Sphere/Cylinder/Plane/Capsule indices
// — used directly as GuiToggleGroup's active index in the config screen — don't shift.
enum class ColliderType { Sphere, Cylinder, Plane, Capsule, None };

// Locks the rotation axis to a world axis so the rotation is a simple 2D spin
// (no general axis-angle machinery needed).
enum class RotationAxis { X, Y, Z, None };

// Which world point's velocity stands in for "the collider's velocity" at a contact (see
// collider_point_velocity): the contacting particle's own position (cheap, approximate under
// rotation), or the closest point on the collider's surface (exact for these analytic shapes).
// Global rather than per-Collider — it's a choice about the contact model itself, shared across
// however many colliders eventually exist, not a property of any one collider's geometry/motion.
enum class ContactPointMode { Particle, Surface };

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

    // Shared constant rotation, applied on top of the translation above, whichever shape is active.
    RotationAxis rotation_axis   = RotationAxis::None;
    Vec3         rotation_origin = Vec3::Zero(); // point the rotation axis passes through
    Real         omega           = 0.0;          // signed angular velocity (rad/s), right-hand rule about rotation_axis
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

inline Real collider_surface_radius(const Collider& c)
{
    switch (c.type)
    {
        case ColliderType::Sphere:   return c.sphere_radius;
        case ColliderType::Cylinder: return c.cylinder_radius;
        case ColliderType::Capsule:  return c.capsule_radius;
        default:                     return 0.0; // Plane has no radius
    }
}

inline Vec3 rotation_axis_vector(RotationAxis axis)
{
    switch (axis)
    {
        case RotationAxis::X: return Vec3::UnitX();
        case RotationAxis::Y: return Vec3::UnitY();
        case RotationAxis::Z: return Vec3::UnitZ();
        default:              return Vec3::Zero();
    }
}

inline Vec3 collider_transform_point(const Collider& c, const Vec3& p, Real time)
{
    Vec3 out = p;
    if (c.rotation_axis != RotationAxis::None && c.omega != 0.0)
    {
        const Eigen::AngleAxis<Real> R(c.omega * time, rotation_axis_vector(c.rotation_axis));
        out = c.rotation_origin + R * (p - c.rotation_origin);
    }
    return out + c.velocity * time;
}

inline Vec3 collider_transform_direction(const Collider& c, const Vec3& d, Real time)
{
    if (c.rotation_axis == RotationAxis::None || c.omega == 0.0) return d;
    const Eigen::AngleAxis<Real> R(c.omega * time, rotation_axis_vector(c.rotation_axis));
    return R * d;
}

inline Vec3 collider_point_velocity(const Collider& c, const Vec3& x, Real time)
{
    if (c.rotation_axis == RotationAxis::None || c.omega == 0.0) return c.velocity;
    const Vec3 omega_vec     = c.omega * rotation_axis_vector(c.rotation_axis);
    const Vec3 current_pivot = collider_transform_point(c, c.rotation_origin, time);
    return c.velocity + omega_vec.cross(x - current_pivot);
}

struct ColliderPose
{
    Vec3 sphere_center;
    Vec3 cylinder_origin, cylinder_axis;
    Vec3 plane_origin,    plane_normal;
    Vec3 capsule_p0,      capsule_p1;
};

// Axis-aligned bounding box for a collider's current pose, used as a cheap per-particle reject
// before the shape's real distance test. Only meaningful for finite shapes (Sphere, Capsule) —
// Cylinder (infinite line) and Plane (infinite sheet) have no useful bound, so `valid` stays false
// and callers should skip the AABB check entirely for those (always fall through to the real test).
struct AABB
{
    Vec3 min   = Vec3::Zero();
    Vec3 max   = Vec3::Zero();
    bool valid = false;
};

inline AABB collider_aabb(const Collider& c, const ColliderPose& pose)
{
    AABB box;
    if (c.type == ColliderType::Sphere)
    {
        box.min   = pose.sphere_center - Vec3::Constant(c.sphere_radius);
        box.max   = pose.sphere_center + Vec3::Constant(c.sphere_radius);
        box.valid = true;
    }
    else if (c.type == ColliderType::Capsule)
    {
        box.min   = pose.capsule_p0.cwiseMin(pose.capsule_p1) - Vec3::Constant(c.capsule_radius);
        box.max   = pose.capsule_p0.cwiseMax(pose.capsule_p1) + Vec3::Constant(c.capsule_radius);
        box.valid = true;
    }
    return box; // Cylinder/Plane/None: valid = false, min/max unused
}

inline bool aabb_contains(const AABB& box, const Vec3& p)
{
    return (p.array() >= box.min.array()).all() && (p.array() <= box.max.array()).all();
}

inline ColliderPose collider_pose_at(const Collider& c, Real time)
{
    ColliderPose pose{};
    switch (c.type)
    {
        case ColliderType::Sphere:
            pose.sphere_center = collider_transform_point(c, c.sphere_center, time);
            break;
        case ColliderType::Cylinder:
            pose.cylinder_origin = collider_transform_point(c, c.cylinder_origin, time);
            pose.cylinder_axis   = collider_transform_direction(c, c.cylinder_axis, time);
            break;
        case ColliderType::Plane:
            pose.plane_origin = collider_transform_point(c, c.plane_origin, time);
            pose.plane_normal = collider_transform_direction(c, c.plane_normal, time);
            break;
        case ColliderType::Capsule:
            pose.capsule_p0 = collider_transform_point(c, c.capsule_p0, time);
            pose.capsule_p1 = collider_transform_point(c, c.capsule_p1, time);
            break;
        case ColliderType::None:
            break;
    }
    return pose;
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

// Which factory builds the sim object: an open rectangular sheet (cloth()) or a closed conical
// skirt (skirt()). Selected in the config screen; AppConfig carries both shapes' params
// simultaneously (like Collider/ColliderType), so switching never lands on the other shape's
// generic defaults. See build_cloth() below for the dispatch.
enum class ClothType { Square, Skirt };

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

// A conical-frustum ("skirt") cloth: `num_rings` circular rings of `particles_per_ring` vertices
// each, stacked along -Y from a pinned `radius_top` ring down to a `radius_bottom` ring `height`
// below it (radii linearly interpolated in between). See diffpd.cpp for the full derivation —
// unlike cloth(), the ring direction wraps around into a closed loop while the height direction
// stays open (no cap at either end), so it reads as a skirt rather than a closed cylinder.
Object skirt(
    Index   particles_per_ring,
    Index   num_rings,
    Real    stiffness     = DEFAULT_STIFFNESS,
    Vec3    origin        = Vec3::Zero(),
    Real    radius_top    = 1.0,
    Real    radius_bottom = 1.5,
    Real    height        = 1.5,
    uint8_t flags         = ClothFlags::ALL,
    Real    m_tot         = 1.0);

// ----------------
//   APP CONFIG
// ----------------

// Pre-fills every shape's params (the literals main() used to leave commented out for the three
// inactive shapes, plus whichever was actually active) so switching collider shape in the config
// screen never lands on Collider's own generic defaults.
inline Collider default_config_collider()
{
    Collider c;
    c.type            = ColliderType::Sphere;
    c.sphere_center   = Vec3(1.0, -1.0, 1.0);
    c.sphere_radius   = 0.5;
    c.cylinder_origin = Vec3(5.0, -6.0, 5.0);
    c.cylinder_axis   = Vec3::UnitX();
    c.cylinder_radius = 1.0;
    c.plane_origin    = Vec3(0.0, -1.4, 0.0);
    c.plane_normal    = Vec3::UnitY();
    c.capsule_p0      = Vec3(5.0, -10.0, 0.0);
    c.capsule_p1      = Vec3(5.0, -5.0, 0.0);
    c.capsule_radius  = 1.0;
    c.velocity        = Vec3::Zero();
    return c;
}

// Every knob main() used to hardcode, now editable live in the pre-run config screen. Default
// member initializers reproduce the old literals exactly, so accepting all defaults and hitting
// Run reproduces the previous hardcoded behavior byte-for-byte.
struct AppConfig
{
    // cloth shape selector — Square uses the width/height/pin_mode/hang_mode block below, Skirt
    // uses the particles_per_ring/num_rings/radius_top/radius_bottom/skirt_height block.
    ClothType cloth_type = ClothType::Square;

    // cloth (shared)
    Real stiffness        = 1.0;
    Real target_stiffness = 2.0;
    Vec3 origin            = Vec3(0.0, 0.0, 0.0);
    Vec3 target_origin     = Vec3(0.0, 0.0, 0.0);
    bool flag_shear        = true;  // ClothFlags::SHEAR   (STRETCH always on, not stored)
    bool flag_bending      = true;  // ClothFlags::BENDING
    Real m_tot              = 0.1;

    // cloth (Square only)
    int         width     = 15;
    int         height    = 15;
    PinMode     pin_mode  = PinMode::ROW;
    HangingMode hang_mode = HangingMode::HORIZONTAL;

    // cloth (Skirt only)
    int  particles_per_ring = 16;
    int  num_rings          = 10;
    Real radius_top         = 1.0;
    Real radius_bottom      = 1.5;
    Real skirt_height       = 1.5;

    // colliders (each entry's four shapes' params live simultaneously, exactly like Collider itself).
    // Fully UI-managed: starts as one default sphere; the config screen's "+"/"-" controls grow or
    // shrink this list (down to empty, disabling contact entirely).
    std::vector<Collider> colliders = { default_config_collider() };

    // Global contact-model choice (applies regardless of which collider shape is active) —
    // see ContactPointMode.
    ContactPointMode contact_point_mode = ContactPointMode::Surface;

    // physics
    Vec3 gravity = Vec3::UnitY() * -9.81;

    // simulation / solver
    int FPS             = 60;
    int frame_substeps  = 1;
    int secs            = 5;
    int n_iters         = 300; // forward PD global-local iterations per step
    int n_iters_adjoint = 300; // backward adjoint-vector iterations per step

    // output
    bool record_on_run = false; // start the screen recorder automatically when Run is clicked

    // gradient check (dphi/dk vs. central finite differences)
    bool run_fd_check = false;
    // one flag per order of magnitude in kFDEpsilonValues (1e-2 .. 1e-10); defaults reproduce the
    // eps set main() used to hardcode ({1e-8, 1e-9, 1e-10}).
    bool fd_eps_selected[9] = { false, false, false, false, false, false, true, true, true };
};

// Dispatches to cloth() or skirt() per cfg.cloth_type, reading each shape's own param block from
// cfg and sharing flag_shear/flag_bending/m_tot between them. `stiffness` and `origin` are passed
// explicitly (not read from cfg) since callers need this for target/guess pairs — and, for the FD
// stiffness check, arbitrary perturbed values — that don't correspond to any single cfg field.
inline Object build_cloth(const AppConfig& cfg, Real stiffness, Vec3 origin)
{
    const uint8_t flags = ClothFlags::STRETCH
                         | (cfg.flag_shear   ? ClothFlags::SHEAR   : 0)
                         | (cfg.flag_bending ? ClothFlags::BENDING : 0);
    switch (cfg.cloth_type)
    {
        case ClothType::Skirt:
            return skirt(cfg.particles_per_ring, cfg.num_rings, stiffness, origin,
                         cfg.radius_top, cfg.radius_bottom, cfg.skirt_height, flags, cfg.m_tot);
        default: // ClothType::Square
            return cloth(cfg.width, cfg.height, stiffness, origin, cfg.pin_mode, cfg.hang_mode,
                         flags, cfg.m_tot);
    }
}

// Order of magnitude choices offered by the FD-check checkboxes, indexed the same way as
// AppConfig::fd_eps_selected (index 0 = 1e-2 ... index 8 = 1e-10).
inline constexpr Real kFDEpsilonValues[9] = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };

// Summary of the loss and adjoint gradients computed for one guess/target pair, handed to the
// playback screen purely for display. dphi_dx0/dphi_dv0 are per-particle (3*num_particles) vectors
// in the actual gradient; here they're collapsed to their column sums — net gradient along each
// world axis, aggregated across every free particle — the same reduction main() already does
// before printing them to stdout (see the Vec3 dphi_dv0/dphi_dx0 locals just above those cout lines).
struct GradientSummary
{
    Real loss     = 0.0;
    Vec3 dphi_dx0 = Vec3::Zero();
    Vec3 dphi_dv0 = Vec3::Zero();
    Real dphi_dk  = 0.0;
};

// Per-step convergence residuals for one guess/target run, handed to the playback screen purely
// for the "how well did each solve converge, over the trajectory" line graphs. target_forward and
// guess_forward are each Tape::forward_residual from their respective forward pass; backward_adjoint
// is BackwardGradContact::residual from the adjoint pass — all three are sized n_steps and indexed
// in forward chronological order.
struct ResidualHistory
{
    std::vector<Real> target_forward;
    std::vector<Real> guess_forward;
    std::vector<Real> backward_adjoint;
};

// Result of one central-difference check of dphi/dk against the analytic gradient (see
// fd_check_contact_stiffness in diffpd.cpp). Shared here (rather than staying private to
// diffpd.cpp) purely as a data type, so the viewer can display FD-check results triggered from the
// playback screen without needing to know about Loss/pd_contact/Object construction itself.
struct FDCheckResult
{
    Real fd;
    Real analytic;
    Real rel_err;
};

// Runs a stiffness FD check for one epsilon per entry of `epsilons`, returning the parallel results.
// diffpd.cpp builds the actual closure (capturing target_tape, gravity, n_iters, the current guess
// stiffness, etc. — see main()); the viewer just calls it with whichever epsilons are selected on
// whichever screen (config or playback) offered the check, without needing to know how it works.
using FDCheckRunner = std::function<std::vector<FDCheckResult>(const std::vector<Real>& epsilons)>;

// ----------------
//    CONTACT
// ----------------

struct Contact
{
    ParticleId particle;
    int        collider_id = -1; // index into the colliders vector this contact was found against
    Vec3       normal; // unit outward contact normal
    Real       inv_r;  // curvature scale (1/dist to the collider's center/axis); 0 for a plane
    bool       active; // set in the backward pass: true if the contact is pressing (d_n < 0)
    Real       d_n;    // set in the backward pass: f_i . normal, cached for the d(normal)/dx term
    Vec3       axis = Vec3::Zero(); // unit collider axis; zero unless the collider is a Cylinder
    Vec3       surface_point = Vec3::Zero(); // closest point on the collider's surface, cached
                                              // alongside normal for ContactPointMode::Surface
    Vec3       v_c = Vec3::Zero(); // collider velocity at this contact point, cached by the forward
                                   // pass so the backward pass sees the same value bit-for-bit
                                   // (differs from collider.velocity whenever the collider rotates)
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

    // Forward-solve convergence diagnostic: the velocity-space fixed-point relative step size at
    // the *last* global-local iteration of each step, ‖v_hat - v‖ / ‖v_hat‖ (see pd_contact) — one
    // entry per simulation step (size == n_steps), independent of the `verbose` flag that gates
    // whether pd_contact also prints a warning when this exceeds its convergence threshold.
    std::vector<Real> forward_residual;

    void clear()
    {
        positions.clear();
        velocities.clear();
        contacts.clear();
        forward_residual.clear();
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
