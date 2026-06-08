#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

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
using ParticleId   = int;
using ConstraintId = int;

using Vec2 = Eigen::Matrix<Real, 2, 1>;
using Vec3 = Eigen::Matrix<Real, 3, 1>;
using Vec4 = Eigen::Matrix<Real, 4, 1>;
using Vec6 = Eigen::Matrix<Real, 6, 1>;

using Mat2 = Eigen::Matrix<Real, 2, 2>;
using Mat3 = Eigen::Matrix<Real, 3, 3>; 
using Mat6 = Eigen::Matrix<Real, 6, 6>;

using RowVec3 = Eigen::Matrix<Real, 1, 3>;
using RowVec6 = Eigen::Matrix<Real, 1, 6>;

using PointsX  = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RealVecX = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

using SparseMat = Eigen::SparseMatrix<Real>;
using Triplet   = Eigen::Triplet<Real>;

constexpr Real BASE_COMPLIANCE = 1e-5;

struct DistanceConstraint;
struct SimulationTape;
struct Object;
struct DistanceCorrection;
struct AdjointState;
struct LossGradients;

using Constraints = std::vector<DistanceConstraint>;
using Positions   = PointsX;
using Velocities  = PointsX;
using InvWeights  = RealVecX;
using AdjointPositions  = Eigen::VectorXd;
using AdjointVelocities = Eigen::VectorXd;

inline bool is_pinned(Real inv_weight) { return inv_weight == 0.0; }

// ----------------
//     OBJECT
// ----------------

struct Object
{
    Positions  x;
    Velocities v;
    InvWeights w;

    Positions  prev_x;

    Constraints constraints;

    Index num_particles() { return x.rows(); }
};

// ----------------
//    FACTORIES
// ----------------

enum class ObjType { CHAIN, CLOTH };

struct ObjectSpec
{
    std::string name;
    std::vector<int> args;
};

namespace make
{
    Object chain(
        Index      n_particles,
        Real       spacing    = 1.0,
        Real       compliance = BASE_COMPLIANCE,
        Vec3       origin     = Vec3::Zero(),
        Vec3       direction  = Vec3::UnitX(),
        bool       pin_first  = true)
    {
        Object obj;

        obj.x.resize(n_particles, 3);
        for (Index i = 0; i < n_particles; ++i)
            obj.x.row(i) = (origin + i * spacing * direction).transpose();

        obj.v      = Velocities::Zero(n_particles, 3);
        obj.prev_x = obj.x;

        obj.w = InvWeights::Ones(n_particles);
        if (pin_first) obj.w(0) = 0.0;

        // Constraints: one distance constraint per adjacent pair
        obj.constraints.reserve(n_particles - 1);
        for (Index i = 0; i < n_particles - 1; ++i)
            obj.constraints.emplace_back(compliance, spacing, i, i + 1);

        return obj;
    }

    Object cloth(
        Index width,
        Index height,
        Real  spacing    = 1.0,
        Real  compliance = BASE_COMPLIANCE,
        Vec3  origin     = Vec3::Zero())
    {
        Object obj;
        const Index N = width * height;

        auto idx = [height](Index i, Index j) { return i * height + j; };

        obj.x.resize(N, 3);
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height; ++j)
                obj.x.row(idx(i, j)) =
                    (origin + Vec3(i * spacing, 0.0, j * spacing)).transpose();

        obj.v      = Velocities::Zero(N, 3);
        obj.prev_x = obj.x;

        obj.w = InvWeights::Ones(N);
        obj.w(idx(0, 0))          = 0.0;
        obj.w(idx(width - 1, 0))  = 0.0;

        // structural: horizontal (along j)
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 1; ++j)
                obj.constraints.emplace_back(compliance, spacing, idx(i, j), idx(i, j + 1));

        // structural: vertical (along i)
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height; ++j)
                obj.constraints.emplace_back(compliance, spacing, idx(i, j), idx(i + 1, j));

        // shear: both diagonals of each cell
        const Real diag = spacing * std::sqrt(Real(2));
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height - 1; ++j)
            {
                obj.constraints.emplace_back(compliance, diag, idx(i, j),     idx(i + 1, j + 1));
                obj.constraints.emplace_back(compliance, diag, idx(i + 1, j), idx(i, j + 1));
            }

        return obj;
    }
 
    ObjType obj_type_from_string(const std::string& name)
    {
        if (name == "chain") return ObjType::CHAIN;
        if (name == "cloth") return ObjType::CLOTH;
        ASSERT(false, "unknown object type: " << name);
        return ObjType::CHAIN;
    }

    Object object(
        const ObjectSpec& spec,
        Real  compliance,
        Vec3  origin,
        Real  spacing = 1.0)
    {
        const ObjType type = obj_type_from_string(spec.name);

        switch (type)
        {
            case ObjType::CHAIN:
                ASSERT(spec.args.size() == 1,
                    "chain expects 1 arg (length), got " << spec.args.size());
                return 
                    make::chain(
                        spec.args[0], 
                        spacing, 
                        compliance, 
                        origin,
                        Vec3::UnitX(), 
                        /*pin_first=*/true);

            case ObjType::CLOTH:
                ASSERT(spec.args.size() == 2,
                    "cloth expects 2 args (width, height), got " << spec.args.size());
                return 
                    make::cloth(
                        spec.args[0], 
                        spec.args[1], 
                        spacing, 
                        compliance, 
                        origin);
        }

        ASSERT(false, "unhandled object type");
        return Object{};
    }

}

// ----------------
//   CONSTRAINT
// ----------------

struct DistanceCorrection
{
    Vec3 dx_p1;
    Vec3 dx_p2;
    Real dlambda;
};

struct DistanceConstraint 
{
    Real compliance = BASE_COMPLIANCE;

    Real rest_length;
    ParticleId p1, p2;

    Real lambda = 0.0;

    Mat6 ddeltax_dx;

    Vec6 dx_dalpha;

    DistanceConstraint(
        Real compliance, 
        Real rest_length, 
        ParticleId p1, 
        ParticleId p2) :
            compliance(compliance), 
            rest_length(rest_length), 
            p1(p1), 
            p2(p2)
    {
    }

    DistanceCorrection compute_correction(const Positions& x, const InvWeights& w, Real dt)
    {

        const Real w1 = w(p1);
        const Real w2 = w(p2);
        const Vec3 x1 = x.row(p1);
        const Vec3 x2 = x.row(p2);

        const Vec3 delta = x1 - x2;
        const Real dist  = delta.norm();
        if (dist < 1e-12) 
        {
            WARNING("distance between particles is less then safe threshold 1e-12");
            return {Vec3::Zero(), Vec3::Zero(), 0.0};
        }

        const Vec3 n = delta / dist;
        const Real C = dist - rest_length;

        const Real alpha_tilde = compliance / (dt * dt);
        const Real w_sum       = w1 + w2;
        if (w_sum < 1e-12) 
        { 
            WARNING("solving constraint between pinned/0 inverse mass particles");
            return {Vec3::Zero(), Vec3::Zero(), 0.0};
        }

        const Real delta_lambda = (-C - alpha_tilde * lambda) / (w_sum + alpha_tilde);

        // gradient (1-iteration)
        const Mat3 nn = n * n.transpose(); 
        const Mat3 P  = (Mat3::Identity() - nn) / dist;
        const Real D  = w_sum + alpha_tilde;

        Mat6 H;
        H << P, -P,
            -P,  P;

        Mat6 JJ;
        JJ << nn, -nn,
             -nn,  nn;

        Mat6 W = Mat6::Zero();
        W.diagonal().head<3>().setConstant(w(p1));
        W.diagonal().tail<3>().setConstant(w(p2));

        const Real C_over_D2 = C / (D * D);

        ddeltax_dx = 
            H * delta_lambda - 
            JJ / D + 
            (Real(2) * C_over_D2) * (JJ * (W * H));

        ddeltax_dx = W * ddeltax_dx;

        dx_dalpha.head<3>() =  w1 * n * C_over_D2;
        dx_dalpha.tail<3>() = -w2 * n * C_over_D2;

        return {
             delta_lambda * w1 * n, /*dx_p1=*/  
            -delta_lambda * w2 * n, /*dx_p2=*/  
             delta_lambda           /*dlambda=*/
        };
    }

    void solve(Positions& x, const InvWeights& w, Real dt)
    {
        const DistanceCorrection corr = compute_correction(x, w, dt);
        x.row(p1) += corr.dx_p1.transpose();
        x.row(p2) += corr.dx_p2.transpose();
        lambda    += corr.dlambda;
    }

    void reset_lambda() { lambda = 0.0; }
};

// ----------------
//   OBJ OUTPUT
// ----------------

void write_obj(const Object& obj, const std::string& path)
{
    std::ofstream file(path);
    ASSERT(file.is_open(), "could not open OBJ file for writing: " << path);

    file << std::fixed << std::setprecision(6);

    for (Index i = 0; i < obj.x.rows(); ++i)
        file << "v " << obj.x(i, 0) << " " << obj.x(i, 1) << " " << obj.x(i, 2) << "\n";

    for (const auto& c : obj.constraints)
        file << "l " << (c.p1 + 1) << " " << (c.p2 + 1) << "\n";
}

std::string frame_path(const std::string& folder, const std::string& prefix, int frame)
{
    std::ostringstream ss;
    ss << folder << "\\" << prefix << "_"
       << std::setw(6) << std::setfill('0') << frame << ".obj";
    return ss.str();
}

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
//   COLLIDER
// ----------------

using CollisionJacobians = std::vector<Mat3>;

struct ProjectResult 
{
    Vec3 x;
    Mat3 J;
    bool active;
};

struct Collider 
{
    virtual ~Collider() = default;
    virtual ProjectResult project(const Vec3& x) const = 0;
};

struct Halfspace : public Collider
{
    Vec3 ori, n;
    Mat3 nn;

    Halfspace(const Vec3& ori_, const Vec3& n_)
        : ori(ori_), n(n_.normalized())
    {
        nn = Mat3::Identity() - n * n.transpose();
    }

    ProjectResult project(const Vec3& x) const override
    {
        const Real phi = (x - ori).dot(n);
        if (phi >= 0.0) return { x, Mat3::Identity(), false };
        return { x - phi * n, nn, true };
    }
};

CollisionJacobians collision_response(Object& obj, const Collider& collider)
{
    const Index N = obj.num_particles();
    CollisionJacobians jacobians(N, Mat3::Identity());

    for (Index i = 0; i < N; ++i)
    {
        if (is_pinned(obj.w(i))) continue;
        const ProjectResult result = collider.project(obj.x.row(i));
        jacobians[i] = result.J;
        obj.x.row(i) = result.x;
    }

    return jacobians;
}

// ----------------
//   JACOBIANS
// ----------------

SparseMat assemble_collision_jacobian(const CollisionJacobians& coll, Index n_particles)
{
    const Index dim = 3 * n_particles;
    std::vector<Triplet> t;
    t.reserve(n_particles * 9);
    for (Index p = 0; p < n_particles; ++p) {
        const Mat3& Jp = coll[p];
        const Index base = 3 * p;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (Jp(i, j) != 0.0)
                    t.emplace_back(base + i, base + j, Jp(i, j));
    }
    SparseMat M(dim, dim);
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

SparseMat assemble_system_jacobian(
    const Constraints& constraints, 
    const CollisionJacobians& coll_jacobians, 
    Index n_particles)
{
    const Index dim = 3 * n_particles;

    std::vector<Triplet> triplets;
    triplets.reserve(constraints.size() * 36);

    for (Index k = 0; k < dim; ++k)
        triplets.emplace_back(k, k, Real(1));

    for (const auto& c : constraints)
    {
        const Index base1 = 3 * c.p1;
        const Index base2 = 3 * c.p2;
        const Index bases[2] = { base1, base2 };

        for (int bi = 0; bi < 2; ++bi)
        for (int bj = 0; bj < 2; ++bj)
        for (int i  = 0; i  < 3; ++i)
        for (int j  = 0; j  < 3; ++j)
        {
            const Real v = c.ddeltax_dx(3 * bi + i, 3 * bj + j);
            if (v != 0.0)
                triplets.emplace_back(bases[bi] + i, bases[bj] + j, v);
        }
    }

    SparseMat A(dim, dim);
    A.setFromTriplets(triplets.begin(), triplets.end());

    SparseMat J_coll = assemble_collision_jacobian(coll_jacobians, n_particles);
    SparseMat J = J_coll * A;
    J.makeCompressed();
    return J;
}

SparseMat assemble_compliance_jacobian(
    const Constraints& constraints, 
    const CollisionJacobians& coll_jacobians, 
    Index n_particles)
{
    const Index n_rows = 3 * n_particles;
    const Index n_cols = Index(constraints.size());

    std::vector<Triplet> triplets;
    triplets.reserve(constraints.size() * 6);

    for (Index ci = 0; ci < n_cols; ++ci)
    {
        const DistanceConstraint& c = constraints[ci];

        const Index base1 = 3 * c.p1;
        const Index base2 = 3 * c.p2;

        for (int k = 0; k < 3; ++k)
        {
            const Real v1 = c.dx_dalpha(k);
            const Real v2 = c.dx_dalpha(3 + k);

            if (v1 != 0.0) triplets.emplace_back(base1 + k, ci, v1);
            if (v2 != 0.0) triplets.emplace_back(base2 + k, ci, v2);
        }
    }

    SparseMat dxdA(n_rows, n_cols);
    dxdA.setFromTriplets(triplets.begin(), triplets.end());
    SparseMat J_coll = assemble_collision_jacobian(coll_jacobians, n_particles);
    return J_coll * dxdA;
}

// ----------------
//   TAPE / HISTORY
// ----------------

struct SimulationTape
{
    std::vector<SparseMat> jacobians;
    std::vector<SparseMat> compliance_jac;
    std::vector<Positions> positions;

    SimulationTape(Index n_steps)
    {
        reserve(n_steps);
    }

    void reserve(Index n_steps)
    {
        jacobians.reserve(n_steps);
        compliance_jac.reserve(n_steps);
        positions.reserve(n_steps);
    }

    void clear()
    {
        jacobians.clear();
        compliance_jac.clear();
        positions.clear();
    }

    void record(SparseMat J, SparseMat dA, Positions x)
    {
        jacobians.push_back(std::move(J));
        compliance_jac.push_back(std::move(dA));
        positions.push_back(std::move(x));
    }

    Index size() const { return Index(jacobians.size()); }
};

// ----------------
//      XPBD
// ----------------

inline void predict(Object& obj, Real dt, Vec3 gravity)
{
    obj.prev_x = obj.x;

    for (Index i = 0; i < obj.x.rows(); ++i)
    {
        if (is_pinned(obj.w(i))) continue;
        obj.x.row(i) += dt * obj.v.row(i) + dt * dt * gravity.transpose();
    }

    for (auto& c : obj.constraints) c.reset_lambda();
}

inline void update_velocities(Object& obj, Real dt)
{
    for (Index i = 0; i < obj.x.rows(); ++i)
    {
        if (is_pinned(obj.w(i))) { obj.v.row(i).setZero(); continue; }
        obj.v.row(i) = (obj.x.row(i) - obj.prev_x.row(i)) / dt;
    }
}

void XPBD_step_gauss_seidel(Object& obj, Real dt, Vec3 gravity, Index n_iter = 1)
{
    predict(obj, dt, gravity);

    for (Index k = 0; k < n_iter; ++k)
        for (auto& c : obj.constraints)
            c.solve(obj.x, obj.w, dt);

    update_velocities(obj, dt);
}

void XPBD_step_jacobi_1iter(Object& obj, Real dt, Vec3 gravity, const Collider& collider, SimulationTape& tape)
{
    predict(obj, dt, gravity);

    {
        Positions dx = Positions::Zero(obj.x.rows(), 3);
        for (auto& c : obj.constraints)
        {
            const DistanceCorrection corr = c.compute_correction(obj.x, obj.w, dt);
            dx.row(c.p1) += corr.dx_p1.transpose();
            dx.row(c.p2) += corr.dx_p2.transpose();
            c.lambda     += corr.dlambda;
        }
        obj.x += dx;
    }

    CollisionJacobians coll_jacobians = collision_response(obj, collider);

    update_velocities(obj, dt);

    tape.record(
        assemble_system_jacobian(obj.constraints, coll_jacobians, (Index) obj.x.rows()),
        assemble_compliance_jacobian(obj.constraints, coll_jacobians, (Index) obj.x.rows()),
        obj.x
    );
}

// ----------------
//      LOSS
// ----------------

struct LossGradients
{
    Real scalar;
    std::vector<Positions> dphi_dx;  // size T, each (N,3)
    std::vector<Positions> dphi_dv;  // size T, each (N,3)
};

LossGradients mse_final(
    const std::vector<Positions>& trajectory,
    const Positions& target)
{
    const Index T = Index(trajectory.size());
    const Index N = trajectory[0].rows();
    const Real  norm_factor = Real(1) / Real(N * 3);  // mean over all entries

    LossGradients out;
    out.dphi_dx.assign(T, Positions::Zero(N, 3));
    out.dphi_dv.assign(T, Positions::Zero(N, 3));

    const Positions diff = trajectory[T-1] - target;
    out.scalar = diff.squaredNorm() * norm_factor;
    out.dphi_dx[T-1] = (Real(2) * norm_factor) * diff;
    // all other dphi_dx[t] and all dphi_dv[t] stay zero

    return out;
}

LossGradients mse_trajectory(
    const std::vector<Positions>& trajectory,
    const std::vector<Positions>& target)
{
    const Index T = Index(trajectory.size());
    const Index N = trajectory[0].rows();
    const Real  norm_factor = Real(1) / Real(T * N * 3);

    LossGradients out;
    out.dphi_dx.resize(T);
    out.dphi_dv.assign(T, Positions::Zero(N, 3));

    Real total = 0;
    for (Index t = 0; t < T; ++t)
    {
        const Positions diff = trajectory[t] - target[t];
        total += diff.squaredNorm();
        out.dphi_dx[t] = (Real(2) * norm_factor) * diff;
    }
    out.scalar = total * norm_factor;
    return out;
}

LossGradients mse_frames_trajectory(
    const std::vector<Positions>& trajectory,
    const std::vector<Positions>& target,
    int frame_step_length)
{
    const Index T = Index(trajectory.size());
    const Index N = trajectory[0].rows();

    const Index n_frames = (T + frame_step_length - 1) / frame_step_length;
    const Real  norm_factor = Real(1) / Real(n_frames * N * 3);

    LossGradients out;
    out.dphi_dx.assign(T, Positions::Zero(N, 3));
    out.dphi_dv.assign(T, Positions::Zero(N, 3));

    Real total = 0;
    for (Index t = 0; t < T; ++t)
    {
        if (t % frame_step_length != 0) continue;
        const Positions diff = trajectory[t] - target[t];
        total += diff.squaredNorm();
        out.dphi_dx[t] = (Real(2) * norm_factor) * diff;
    }
    out.scalar = total * norm_factor;
    return out;
}

using LossSpec = ObjectSpec; 

LossGradients build_loss(
    const LossSpec& loss_spec,
    const std::vector<Positions>& guess_traj,
    const std::vector<Positions>& target_traj,
    int sim_rate)
{
    if (loss_spec.name == "mse_final_position")
        return mse_final(guess_traj, target_traj.back());

    if (loss_spec.name == "mse_full_trajectory")
        return mse_trajectory(guess_traj, target_traj);

    if (loss_spec.name == "mse_frames_trajectory")
    {
        ASSERT(loss_spec.args.size() == 1,
            "mse_frames_trajectory expects 1 arg (fps), got " << loss_spec.args.size());
        return mse_frames_trajectory(guess_traj, target_traj, sim_rate / loss_spec.args[0]);
    }

    ASSERT(false, std::string("unknown loss: ") + loss_spec.name);
    return LossGradients{};
}

// ----------------
//    ADJOINT
// ----------------

struct AdjointState
{
    AdjointPositions  x_hat;  // size 3N
    AdjointVelocities v_hat;  // size 3N
};

inline Eigen::VectorXd flatten(const Positions& X)
{
    return Eigen::Map<const Eigen::VectorXd>(X.data(), X.size());
}

inline Positions unflatten(const Eigen::VectorXd& v, Index N)
{
    Positions X(N, 3);
    Eigen::Map<Eigen::VectorXd>(X.data(), X.size()) = v;
    return X;
}

std::vector<AdjointState> backward_explicit_adjoint(
    const SimulationTape& tape,
    const LossGradients& loss,
    Real dt)
{
    const Index T = tape.size();
    // const Index N   = tape.positions[0].rows();
    // const Index dim = 3 * N;

    std::vector<AdjointState> adj(T + 1);

    // seed: loss term on the final state x_T = positions[T-1]
    adj[T].x_hat = flatten(loss.dphi_dx[T-1]);
    adj[T].v_hat = flatten(loss.dphi_dv[T-1]);

    for (Index k = T - 1; k >= 1; --k) {
        const SparseMat JkT = SparseMat(tape.jacobians[k].transpose());
        const auto& xh = adj[k+1].x_hat;  const auto& vh = adj[k+1].v_hat;
        const Eigen::VectorXd Jx = JkT * xh, Jv = JkT * vh;
        adj[k].x_hat = 
            Jx + (1/dt)*Jv - (1/dt)*vh + flatten(loss.dphi_dx[k-1]);
        adj[k].v_hat = 
            dt*Jx + Jv + flatten(loss.dphi_dv[k-1]);
    }

    {
        const SparseMat J0T = SparseMat(tape.jacobians[0].transpose());
        const auto& xh = adj[1].x_hat; const auto& vh = adj[1].v_hat;
        const Eigen::VectorXd Jx = J0T*xh, Jv = J0T*vh;
        adj[0].x_hat = Jx + (1/dt)*Jv - (1/dt)*vh;
        adj[0].v_hat = dt*Jx + Jv;
    }

    // implementation in case ddeltax_dx
    // adj[t].x_hat = 
    //     xh_next               + 
    //     JtT_xh                + 
    //     (Real(1)/dt) * JtT_vh + 
    //     flatten(loss.dphi_dx[t]);
    // adj[t].v_hat = 
    //     dt * xh_next + 
    //     dt * JtT_xh  + 
    //     vh_next      + 
    //     JtT_vh       + 
    //     flatten(loss.dphi_dv[t]);

    return adj;
}

std::vector<AdjointState> backward_implicit_adjoint(
    const SimulationTape& tape,
    const LossGradients& loss,
    Real dt)
{
    const Index T   = tape.size();
    const Index N   = tape.positions[0].rows();
    const Index dim = 3 * N;

    std::vector<AdjointState> adj(T);

    adj[T-1].x_hat = flatten(loss.dphi_dx[T-1]);
    adj[T-1].v_hat = flatten(loss.dphi_dv[T-1]);

    SparseMat I(dim, dim);
    I.setIdentity();

    Eigen::SparseLU<SparseMat> solver;

    for (Index t = T - 2; t >= 0; --t)
    {
        const SparseMat& Jt = tape.jacobians[t];

        const SparseMat A = I - Jt;

        solver.compute(A);

        ASSERT(
            solver.info() == Eigen::Success, 
            std::string("SparseLU factorization failed at t=") << t);

        const Eigen::VectorXd& xh_next = adj[t+1].x_hat;
        const Eigen::VectorXd& vh_next = adj[t+1].v_hat;

        const Eigen::VectorXd rhs =
            Real(2) * xh_next
            - (Real(1)/dt) * vh_next
            + flatten(loss.dphi_dx[t]);

        adj[t].x_hat = solver.solve(rhs);

        ASSERT(
            solver.info() == Eigen::Success, 
            std::string("SparseLU solve failed at t=") << t);

        adj[t].v_hat = dt * xh_next + flatten(loss.dphi_dv[t]);
    }

    return adj;
}

// ----------------
//    GRADIENT
// ----------------

Eigen::VectorXd compute_dphi_dcompliance(
    const SimulationTape& tape,
    const LossGradients& loss,
    Real dt)
{
    const Index T = tape.size();
    const Index m = tape.compliance_jac[0].cols();

    const std::vector<AdjointState> adj = backward_explicit_adjoint(tape, loss, dt);

    Eigen::VectorXd dphi_dA = Eigen::VectorXd::Zero(m);

    for (Index k = 0; k < T; ++k)
    {
        const SparseMat& dxdA = tape.compliance_jac[k];

        const Eigen::VectorXd combined =
            adj[k+1].x_hat + (Real(1)/dt) * adj[k+1].v_hat;

        dphi_dA += dxdA.transpose() * combined;
    }

    // dphi_dA += dphi_dA_direct;

    return dphi_dA;
}

// ----------------
//    CONFIG
// ----------------

using ExperimentSpec = ObjectSpec; 


ObjectSpec parse_object_spec(const std::string& spec)
{
    ObjectSpec out;

    // Accept either "name" (no args) or "name(a, b, ...)".
    const auto lp = spec.find('(');

    out.name = spec.substr(0, lp);   // lp == npos -> whole string
    out.name.erase(0, out.name.find_first_not_of(" \t"));
    out.name.erase(out.name.find_last_not_of(" \t") + 1);

    if (lp != std::string::npos)
    {
        const auto rp = spec.find(')');
        ASSERT(rp != std::string::npos && rp > lp,
               "spec has '(' but no matching ')': " << spec);

        std::stringstream ss(spec.substr(lp + 1, rp - lp - 1));
        std::string token;
        while (std::getline(ss, token, ','))
            out.args.push_back(std::stoi(token));
    }

    return out;
}

struct Config
{
    std::unordered_map<std::string, std::string> values;

    Real get_real(const std::string& key) const
    {
        auto it = values.find(key);
        ASSERT(it != values.end(), "missing config key: " << key);
        return std::stod(it->second);
    }

    int get_int(const std::string& key) const
    {
        auto it = values.find(key);
        ASSERT(it != values.end(), "missing config key: " << key);
        return std::stoi(it->second);
    }

    Vec3 get_vec3(const std::string& key) const
    {
        auto it = values.find(key);
        ASSERT(it != values.end(), "missing config key: " << key);

        std::string s;
        for (char ch : it->second)
            if (ch != '(' && ch != ')')
                s += ch;

        std::stringstream ss(s);
        Vec3 v;
        std::string token;
        for (int i = 0; i < 3; ++i)
        {
            ASSERT(std::getline(ss, token, ','),
                   "vec3 key '" << key << "' needs 3 comma-separated values, got: " << it->second);
            v(i) = std::stod(token);
        }
        return v;
    }

    bool get_bool(const std::string& key) const
    {
        auto it = values.find(key);
        ASSERT(it != values.end(), "missing config key: " << key);

        std::string s = it->second;
        // for (char& ch : s) ch = std::tolower(static_cast<unsigned char>(ch));

        if (s == "true")  return true;
        if (s == "false") return false;

        ASSERT(false, "bool key '" << key << "' must be true/false, got: " << it->second);
        return false;
    }

    ObjectSpec get_object(const std::string& key) const
    {
        auto it = values.find(key);
        ASSERT(it != values.end(), "missing config key: " << key);
        return parse_object_spec(it->second);
    }
};

Config load_config(const std::string& path)
{
    std::ifstream file(path);
    ASSERT(file.is_open(), "could not open config file: " << path);

    Config cfg;
    std::string line;
    while (std::getline(file, line))
    {
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;

        if (line[first] == '#' || line[first] == ';') continue;
        if (line[first] == '/' && first + 1 < line.size() && line[first + 1] == '/') continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        auto trim = [](std::string s) 
        {
            const auto a = s.find_first_not_of(" \t\r");
            const auto b = s.find_last_not_of(" \t\r");
            return (a == std::string::npos) ? std::string{} : s.substr(a, b - a + 1);
        };

        cfg.values[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return cfg;
}

// ----------------
//     OUTPUT
// ----------------

void print_positions(const std::string& label, const Positions& x) 
{
    auto num_particles = x.rows();
    std::cout << label << " = [";
    for (Index i = 0; i < num_particles; ++i)
        std::cout << "("
                << x(i, 0) << ", "
                << x(i, 1) << ", "
                << x(i, 2) << ")" 
                << (i != num_particles-1 ? ", " : " ");
    std::cout << "]\n";
}

void print_vector(const std::string& label, const Eigen::VectorXd& v)
{
    std::cout << label << " = [";
    for (Index i = 0; i < v.size(); ++i)
        std::cout << v(i) << (i + 1 < v.size() ? ", " : "");
    std::cout << "]\n";
}

// ----------------
//    CONTEXT
// ----------------

struct ExperimentContext
{
    ObjectSpec     obj_spec;
    ExperimentSpec exp_spec;
    LossSpec       loss_spec;

    Real compliance;
    Real target_compliance;

    Vec3 offset;
    Vec3 target_offset;
    Vec3 gravity;

    Halfspace ground;

    int   sim_rate;
    int   frame_step_length;
    Index n_steps;
    Real  dt;

    bool        export_obj;
    std::string anim_folder;
};

struct SimResult
{
    Object         obj;
    SimulationTape tape;
};

SimResult run_sim(const ExperimentContext& ctx, Real compliance, const Vec3& offset, const std::string& prefix)
{
    Object obj = make::object(ctx.obj_spec, compliance, offset);
    SimulationTape tape(ctx.n_steps);

    int frame = 0;
    for (Index step = 0; step < ctx.n_steps; ++step)
    {
        if (ctx.export_obj && step % ctx.frame_step_length == 0)
            write_obj(obj, frame_path(ctx.anim_folder, prefix, frame++));

        XPBD_step_jacobi_1iter(obj, ctx.dt, ctx.gravity, ctx.ground, tape);
    }

    return { std::move(obj), std::move(tape) };
}

// Shared forward stage for the gradient experiments: target sim, guess sim,
// loss, and final-position print. Returns both tapes and the loss.
struct InverseForward
{
    SimResult     target;
    SimResult     guess;
    LossGradients loss;
};

InverseForward inverse_forward(const ExperimentContext& ctx)
{
    SimResult target = run_sim(ctx, ctx.target_compliance, ctx.target_offset, "target");
    SimResult guess  = run_sim(ctx, ctx.compliance,        ctx.offset,        "guess");

    LossGradients loss = build_loss(
        ctx.loss_spec, guess.tape.positions, target.tape.positions, ctx.sim_rate);

    std::cout << "\n=== Final Positions ===\n";
    print_positions("pos_final", target.obj.x);
    print_positions("pos_guess", guess.obj.x);

    return { std::move(target), std::move(guess), std::move(loss) };
}

// ----------------
//   EXPERIMENTS
// ----------------

void experiment_single_step_jacobian(const ExperimentContext& ctx)
{
    ASSERT(ctx.exp_spec.args.size() == 1,
        "single_step_jacobian expects 1 arg (step), got " << ctx.exp_spec.args.size());
    const Index step_index = ctx.exp_spec.args[0];
    ASSERT(step_index >= 1 && step_index <= ctx.n_steps,
        "step must be in [1, " << ctx.n_steps << "], got " << step_index);

    SimResult sim = run_sim(ctx, ctx.compliance, ctx.offset, "obj");

    const SparseMat& J = sim.tape.jacobians[step_index - 1];

    std::cout << "=== d x^+ / d x^-  at update " << step_index << " / " << ctx.n_steps << " ===\n";
    std::cout << "Frobenius norm: " << J.norm() << "\n";
}

void experiment_compliance_gradient(const ExperimentContext& ctx)
{
    InverseForward fwd = inverse_forward(ctx);

    const Eigen::VectorXd dphi_dalpha      = compute_dphi_dcompliance(fwd.guess.tape, fwd.loss, ctx.dt);
    const Eigen::VectorXd dphi_dcompliance = dphi_dalpha / (ctx.dt * ctx.dt);

    std::cout << "\n=== Compliance gradient ===\n";
    print_vector("dL_dalpha", dphi_dcompliance);

    std::cout << "\ndL/dcompliance sum:  " << dphi_dcompliance.sum() << "\n";
    std::cout << "dL/dcompliance mean: "   << dphi_dcompliance.mean() << "\n";
}

void experiment_x0_gradient(const ExperimentContext& ctx)
{
    InverseForward fwd = inverse_forward(ctx);

    const std::vector<AdjointState> adj = backward_explicit_adjoint(fwd.guess.tape, fwd.loss, ctx.dt);

    const Eigen::VectorXd& dL_dx0 = adj[0].x_hat;
    const Index dim               = dL_dx0.size();
    const Index num_particles     = fwd.target.obj.num_particles();

    std::cout << "=== d loss / d x0  (" << num_particles << " particles, " << dim << " dims) ===\n";
    std::cout << "loss: " << fwd.loss.scalar << "\n";
    print_vector("dL_dx0", dL_dx0);
}

// ----------------
//      MAIN
// ----------------

int main(int argc, char** argv)
{
    const std::filesystem::path proj_root =
        std::filesystem::path(__FILE__).parent_path().parent_path();

    const std::filesystem::path config_rel =
        (argc > 1) ? std::filesystem::path(argv[1])
                   : std::filesystem::path("src") / "param.conf";

    const Config cfg = load_config((proj_root / config_rel).string());

    const int sim_rate  = cfg.get_int("sim_rate");
    const int fps       = cfg.get_int("fps");
    const int n_seconds = cfg.get_int("n_seconds");

    const ExperimentContext ctx {
        cfg.get_object("obj"),
        cfg.get_object("experiment"),
        cfg.get_object("loss"),
        cfg.get_real("compliance"),
        cfg.get_real("target_compliance"),
        cfg.get_vec3("offset"),
        cfg.get_vec3("target_offset"),
        cfg.get_vec3("gravity"),
        Halfspace(cfg.get_vec3("ground_ori"), cfg.get_vec3("ground_normal")),
        sim_rate,
        sim_rate / fps,            // frame_step_length
        sim_rate * n_seconds,      // n_steps
        1.0 / (Real)sim_rate,      // dt
        cfg.get_bool("export_obj"),
        (proj_root / "animation").string()
    };

    std::cout << std::scientific << std::setprecision(8);

    if (ctx.export_obj) clear_folder(ctx.anim_folder);

    const std::string& name = ctx.exp_spec.name;
    if      (name == "single_step_jacobian") experiment_single_step_jacobian(ctx);
    else if (name == "compliance_gradient")  experiment_compliance_gradient(ctx);
    else if (name == "x0_gradient")          experiment_x0_gradient(ctx);
    else ASSERT(false, std::string("unknown experiment: ") + name);

    return 0;
}

// ----------------
//   TESTS
// ----------------

struct SingleConstraintResult
{
    Vec3 dx_p1, dx_p2;
    Vec3 dxi_dalpha, dxj_dalpha;
};

SingleConstraintResult single_constraint(
    const Vec3& x1, const Vec3& x2,
    Real w1, Real w2,
    Real rest, Real alpha_tilde)
{
    const Vec3 delta = x1 - x2;
    const Real dist  = delta.norm();
    const Vec3 n     = delta / dist;
    const Real C     = dist - rest;

    const Real D    = w1 + w2 + alpha_tilde;
    const Real dlam = -C / D;

    const Real C_over_D2 = C / (D * D);

    SingleConstraintResult r;
    r.dx_p1 =  dlam * w1 * n;
    r.dx_p2 = -dlam * w2 * n;
    r.dxi_dalpha =  w1 * n * C_over_D2;
    r.dxj_dalpha = -w2 * n * C_over_D2;
    return r;
}

void run_case(
    const char* name,
    const Vec3& x1, const Vec3& x2,
    Real w1, Real w2, 
    Real rest, Real alpha_tilde)
{
    const SingleConstraintResult r =
        single_constraint(x1, x2, w1, w2, rest, alpha_tilde);

    const Vec3 x1_new = x1 + r.dx_p1;
    const Vec3 x2_new = x2 + r.dx_p2;

    std::printf("\n=== %s ===\n", name);
    std::printf("  inputs: x1=(%.1f,%.1f,%.1f) x2=(%.1f,%.1f,%.1f) "
                "w1=%.1f w2=%.1f rest=%.2f alpha=%.6g\n",
                x1(0), x1(1), x1(2), x2(0), x2(1), x2(2),
                w1, w2, rest, alpha_tilde);
    std::printf("  x1_new = (%.10f, %.10f, %.10f)\n", x1_new(0), x1_new(1), x1_new(2));
    std::printf("  x2_new = (%.10f, %.10f, %.10f)\n", x2_new(0), x2_new(1), x2_new(2));
    std::printf("  dx1/dalpha = (%.10e, %.10e, %.10e)\n",
                r.dxi_dalpha(0), r.dxi_dalpha(1), r.dxi_dalpha(2));
    std::printf("  dx2/dalpha = (%.10e, %.10e, %.10e)\n",
                r.dxj_dalpha(0), r.dxj_dalpha(1), r.dxj_dalpha(2));
}

void run_all_tests()
{
    run_case("Case 1: axial stretch, equal mass",
             Vec3(0,0,0), Vec3(1.5,0,0), 1.0, 1.0, 1.0, 0.1);

    run_case("Case 2: axial compression",
             Vec3(0,0,0), Vec3(0.5,0,0), 1.0, 1.0, 1.0, 0.1);

    run_case("Case 3: pinned p1",
             Vec3(0,0,0), Vec3(1.5,0,0), 0.0, 1.0, 1.0, 0.1);

    run_case("Case 4: asymmetric mass",
             Vec3(0,0,0), Vec3(1.5,0,0), 2.0, 0.5, 1.0, 0.1);

    run_case("Case 5: diagonal 3D",
             Vec3(0,0,0), Vec3(1.0,1.0,1.0), 1.0, 1.0, 1.0, 0.1);

    run_case("Case 6: near-rigid (small alpha)",
             Vec3(0,0,0), Vec3(1.2,0,0), 1.0, 1.0, 1.0, 1e-4);
}

int main_test()
{
    run_all_tests();
    return 0;
}
