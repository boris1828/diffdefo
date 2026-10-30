#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <functional>
#include <optional>
#include <iomanip>

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
//  PARAMETER SLOTS
// ----------------

// Which constraint family a spring belongs to: selects its stiffness and its dphi/dk slot.
// Shared by every elastic energy: a new energy picks one of these slots for its parameter.
enum class SpringKind : std::uint8_t { Stretch, Shear, Bend };

// One stiffness (or one dphi/dk) per SpringKind; unified stiffness is all three equal.
struct Stiffness
{
    Real stretch = 0.0, shear = 0.0, bend = 0.0;

    static Stiffness uniform(Real k) { return { k, k, k }; }
    Real& operator[](SpringKind kind)
    {
        return kind == SpringKind::Stretch ? stretch : kind == SpringKind::Shear ? shear : bend;
    }
    Stiffness& operator+=(const Stiffness& o) { stretch += o.stretch; shear += o.shear; bend += o.bend; return *this; }
    Real sum() const { return stretch + shear + bend; }
};
