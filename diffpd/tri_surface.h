#pragma once

// Triangle-mesh surface generation (GL-free, header-only). A surface is a flat 2D "chart" Omega plus a map to 3D;
// generate() samples the chart's boundary in 1D, fills the interior with Poisson-disk points, relaxes and
// Delaunay-triangulates them and lifts to 3D. A closed surface (cone, cylinder) is a chart identified with itself under a
// shift (a "seam"): the seam is not a boundary, sampling/Delaunay/smoothing run modulo the shift with ghost copies near it,
// and dropping the ghost index of a triangle corner is the weld. Charts without a seam leave the seam methods at their defaults.
// ConeFrustum (closed, seam), Rectangle (flat, no seam) and PatternChart (any closed line/Bezier contour, no seam) exist today; see diffpd/CLAUDE.md "Triangle-mesh surfaces" for how to add another chart.
// Reference algorithm: prototypes/skirt_mesh_periodic_proto.py.

#include "diffpd_core.h"

#include <delaunator.hpp>

#include <map>
#include <random>
#include <set>
#include <stdexcept>

// Failures a user can reach (a bad pattern file, an h that does not fit the shape) throw instead of aborting, so a UI can show them.
#define MESHGEN_CHECK(cond, message) \
    do { \
        if (!(cond)) { std::ostringstream _oss; _oss << "meshgen: " << message; throw std::runtime_error(_oss.str()); } \
    } while (0)

namespace meshgen {

using Vec2 = Eigen::Matrix<Real, 2, 1>;

constexpr Real kTwoPi = 6.283185307179586476925286766559;

struct BoundaryCurve { int id; bool closed; std::vector<Vec2> p; }; // `id`: surface-defined, used to pick the pinned vertices

class Chart
{
public:
    virtual ~Chart() = default;

    // The fundamental domain Omega, open at the free edges and half-open at a seam (each point class counted once).
    virtual bool contains(const Vec2& p) const = 0;
    virtual void bounds(Vec2& lo, Vec2& hi) const = 0;            // AABB of Omega (for the Poisson grid)
    // Free edges only, sampled in 1D at spacing ~h; NO seam lines and no duplicated end point. Consecutive points of a curve
    // must be adjacent along it (the core checks they end up as mesh edges; `closed` adds the last -> first link).
    virtual std::vector<BoundaryCurve> sample_boundary(Real h) const = 0;
    // Relative to the surface origin. Must be seam-periodic: to3d(shift(p, k)) == to3d(p) (wrap inside if needed).
    virtual Vec3 to3d(const Vec2& p) const = 0;
    virtual Vec3 outward(const Vec3& p) const = 0;                // any vector on the side the normals should face
    virtual Real area() const = 0;                                // exact surface area (validation only)
    virtual int  euler_characteristic() const = 0;                // 0 tube/ring, 1 disk (validation only)
    virtual int  n_boundary_curves() const = 0;                   // curve ids [0, n) are free edges

    // Seam (defaults = none). shift(p, k) is the image of p k periods away, wrap() brings a point into Omega, seam_gap()
    // gives the distance to the two seam lines (lo side: images at +1 appear beyond hi; hi side: images at -1 appear below lo).
    virtual bool periodic() const { return false; }
    virtual Vec2 shift(const Vec2& p, int) const { return p; }
    virtual Vec2 wrap(const Vec2& p) const { return p; }
    virtual void seam_gap(const Vec2&, Real& to_lo, Real& to_hi) const { to_lo = to_hi = 1e300; }
};

struct Params { Real h = 0.07; Real beta = 0.85; int smooth_iters = 5; int k = 30; uint64_t seed = 0; };

struct TriSurface
{
    std::vector<Vec3>                   pos;         // 3D vertices (relative to origin)
    std::vector<Vec2>                   uv;          // chart coordinates of each vertex (inside Omega)
    std::vector<int>                    curve;       // per vertex: boundary curve id, -1 = interior
    std::vector<std::vector<Index>>     boundary;    // per chart boundary curve (in sample_boundary order): its vertex ids in order, shared corners included
    std::vector<std::array<Index, 3>>   tris;        // consistently oriented
    std::vector<std::pair<Index,Index>> edges;       // unique, a < b
    std::vector<Real>                   vertex_area; // lumped: sum of incident triangle areas / 3
};

struct Report
{
    Index vertices = 0, triangles = 0, edges = 0;
    int   euler = 0, open_edges = 0, bad_open_edges = 0, nonmanifold = 0, unused_vertices = 0, interior_on_seam = 0;
    Real  min_angle_deg = 0, mean_angle_deg = 0;
    Real  edge_min = 0, edge_mean = 0, edge_max = 0; // in units of h
    Real  area_error = 0;                            // mesh area / exact area - 1
    bool  structure_ok = false;                      // closed manifold with the chart's topology: generate() asserts this
    bool  quality_ok = false;                        // area < 3%, min angle > 15 deg, min edge > 0.4 h, nothing on the seam
};

// ---- chart: cut cone ----------------------------------------------------------------------------------------------
// Top ring (r_top) at y=0, bottom ring (r_bot) at y=-H, hanging -Y. Unrolled to an annular sector [lo,hi] x [0,theta) with its
// apex at the origin; the seam is the rotation by theta about the apex. Curves: 0 top ring, 1 bottom ring (both closed, no seam
// curves). A cylinder (r_top == r_bot) has its apex at infinity and no finite unrolling: it is rejected.
class ConeFrustum : public Chart
{
public:
    ConeFrustum(Real r_top, Real r_bot, Real H) : r_top_(r_top), r_bot_(r_bot), H_(H)
    {
        MESHGEN_CHECK(std::abs(r_bot - r_top) >= 1e-3, "ConeFrustum: radii must differ by at least 1e-3 (a cylinder has no cone unrolling)");
        MESHGEN_CHECK(r_top > 0 && r_bot > 0 && H > 0, "ConeFrustum: radii and height must be positive");
        const Real dr = r_bot - r_top;
        L_     = std::hypot(H, dr);
        s_     = std::abs(dr) / L_;
        rho_t_ = r_top / s_;
        rho_b_ = r_bot / s_;
        lo_    = std::min(rho_t_, rho_b_);
        hi_    = std::max(rho_t_, rho_b_);
        theta_ = kTwoPi * s_;
    }

    bool contains(const Vec2& p) const override
    {
        constexpr Real eps = 1e-9;
        Real rho, phi;
        polar(p, rho, phi);
        return rho > lo_ + eps && rho < hi_ - eps && phi >= 0 && phi < theta_;
    }

    void bounds(Vec2& lo, Vec2& hi) const override
    {
        lo = Vec2::Constant(1e300);
        hi = Vec2::Constant(-1e300);
        auto grow = [&](Real rho, Real phi)
        {
            const Vec2 p(rho * std::cos(phi), rho * std::sin(phi));
            lo = lo.cwiseMin(p); hi = hi.cwiseMax(p);
        };
        for (Real rho : { lo_, hi_ }) { grow(rho, 0.0); grow(rho, theta_); }
        for (int q = 1; q * kTwoPi / 4 < theta_; ++q) grow(hi_, q * kTwoPi / 4); // arc extremes on the axes
    }

    std::vector<BoundaryCurve> sample_boundary(Real h) const override
    {
        std::vector<BoundaryCurve> out;
        int id = 0;
        for (Real rho : { rho_t_, rho_b_ })
        {
            const int n = (int)std::lround(theta_ * rho / h);
            MESHGEN_CHECK(n >= 6, "ConeFrustum: h = " << h << " is too large, fewer than 6 samples on a ring; lower h");
            BoundaryCurve c{ id++, true, {} };
            for (int i = 0; i < n; ++i) // i = n is the image of i = 0: not added
            {
                const Real phi = theta_ * Real(i) / Real(n);
                c.p.push_back(Vec2(rho * std::cos(phi), rho * std::sin(phi)));
            }
            out.push_back(std::move(c));
        }
        return out;
    }

    Vec3 to3d(const Vec2& p) const override
    {
        Real rho, phi;
        polar(p, rho, phi);
        phi -= theta_ * std::floor(phi / theta_); // phi / s_ must lie in [0, 2 pi): atan2's branch is not periodic in it
        const Real tau = (rho - rho_t_) / (rho_b_ - rho_t_);
        return Vec3(rho * s_ * std::cos(phi / s_), -H_ * tau, rho * s_ * std::sin(phi / s_));
    }

    Vec3 outward(const Vec3& p) const override { return Vec3(p.x(), 0.0, p.z()); }
    Real area() const override { return kTwoPi / 2 * (r_top_ + r_bot_) * L_; }
    int  euler_characteristic() const override { return 0; }
    int  n_boundary_curves() const override { return 2; }

    bool periodic() const override { return true; }
    Vec2 shift(const Vec2& p, int k) const override
    {
        const Real a = k * theta_, c = std::cos(a), sn = std::sin(a);
        return Vec2(c * p.x() - sn * p.y(), sn * p.x() + c * p.y());
    }
    Vec2 wrap(const Vec2& p) const override
    {
        Real rho, phi;
        polar(p, rho, phi);
        return shift(p, -(int)std::floor(phi / theta_));
    }
    void seam_gap(const Vec2& p, Real& to_lo, Real& to_hi) const override
    {
        Real rho, phi;
        polar(p, rho, phi);
        to_lo = rho * phi;
        to_hi = rho * (theta_ - phi);
    }

private:
    // phi is taken in a window centred on the sector, NOT [0, 2 pi): a point just below the seam must stay slightly
    // negative so that wrap() sends it to theta - eps.
    void polar(const Vec2& p, Real& rho, Real& phi) const
    {
        rho = p.norm();
        phi = std::fmod(std::atan2(p.y(), p.x()) - theta_ / 2 + kTwoPi / 2 + 2 * kTwoPi, kTwoPi) - kTwoPi / 2 + theta_ / 2;
    }

    Real r_top_, r_bot_, H_;
    Real L_, s_, rho_t_, rho_b_, lo_, hi_, theta_;
};

// ---- chart: flat rectangle ----------------------------------------------------------------------------------------
// Open box [0,W] x [0,H] in the plane, no seam (every seam method keeps its default). Chart (u,v) -> 3D matches cloth():
// horizontal (u, 0, v), vertical (u, -v, 0), so the v = 0 edge is the top row either way.
// Curves: the four edges (open, each listing its end points) and the top-left / top-right corners as one-point curves emitted
// FIRST, so they own those two vertices (first writer wins in generate()). Pin selection by curve id: row = {Top, CornerTL,
// CornerTR}, corners = {CornerTL, CornerTR}, none = {}.
class Rectangle : public Chart
{
public:
    enum Curve { Top, Right, Bottom, Left, CornerTL, CornerTR }; // Top: v=0, Right: u=W, Bottom: v=H, Left: u=0

    Rectangle(Real W, Real H, bool vertical) : W_(W), H_(H), vertical_(vertical)
    {
        MESHGEN_CHECK(W > 0 && H > 0, "Rectangle: width and height must be positive");
    }

    bool contains(const Vec2& p) const override
    {
        constexpr Real eps = 1e-9;
        return p.x() > eps && p.x() < W_ - eps && p.y() > eps && p.y() < H_ - eps;
    }

    void bounds(Vec2& lo, Vec2& hi) const override { lo = Vec2(0.0, 0.0); hi = Vec2(W_, H_); }

    std::vector<BoundaryCurve> sample_boundary(Real h) const override
    {
        std::vector<BoundaryCurve> out;
        out.push_back({ CornerTL, false, { Vec2(0.0, 0.0) } });
        out.push_back({ CornerTR, false, { Vec2(W_, 0.0) } });
        auto edge = [&](int id, const Vec2& a, const Vec2& b)
        {
            const int n = (int)std::lround((b - a).norm() / h);
            MESHGEN_CHECK(n >= 3, "Rectangle: h = " << h << " is too large, fewer than 3 segments on an edge; lower h");
            BoundaryCurve c{ id, false, {} };
            for (int i = 0; i <= n; ++i) c.p.push_back(a + (b - a) * (Real(i) / Real(n))); // both ends: corners dedup in generate()
            out.push_back(std::move(c));
        };
        edge(Top,    Vec2(0.0, 0.0), Vec2(W_, 0.0));
        edge(Right,  Vec2(W_, 0.0),  Vec2(W_, H_));
        edge(Bottom, Vec2(0.0, H_),  Vec2(W_, H_));
        edge(Left,   Vec2(0.0, 0.0), Vec2(0.0, H_));
        return out;
    }

    Vec3 to3d(const Vec2& p) const override { return vertical_ ? Vec3(p.x(), -p.y(), 0.0) : Vec3(p.x(), 0.0, p.y()); }
    Vec3 outward(const Vec3&) const override { return vertical_ ? Vec3::UnitZ() : Vec3::UnitY(); }
    Real area() const override { return W_ * H_; }
    int  euler_characteristic() const override { return 1; }
    int  n_boundary_curves() const override { return CornerTR + 1; }

private:
    Real W_, H_;
    bool vertical_;
};

// ---- chart: pattern contour ---------------------------------------------------------------------------------------
// A closed contour of straight lines and cubic Beziers (sewing-pattern piece, see clothcreate/FORMAT.md), in image pixels (y down).
// The chart is metres, y up, centred on the contour's bounding box: uv = scale * (x - cx, -(y - cy)); to3d = (u, v, 0), facing +Z.
// Every segment is flattened ONCE (de Casteljau until the control polygon is within flat_tol of the chord) and the same polyline feeds
// the inside test and the boundary samples, so a sample can never fall outside the polygon. Curves: one per input segment, id = the
// segment's index in the input (stable, for pinning / later sewing); zero-length segments emit nothing. Boundary spacing is exactly
// uniform inside a segment (n = round(L/h), segment ends are always samples, so corners stay sharp).
struct PatternSeg { bool curve; Vec2 c1, c2, to; }; // line: only `to`
struct PatternShape { Vec2 start; bool closed; std::vector<PatternSeg> segs; };

class PatternChart : public Chart
{
public:
    PatternChart(const PatternShape& shape, Real scale, Real flat_tol) : n_segs_((int)shape.segs.size())
    {
        MESHGEN_CHECK(shape.closed, "PatternChart: the shape is not closed");
        MESHGEN_CHECK(scale > 0 && flat_tol > 0, "PatternChart: scale and flat_tol must be positive");
        MESHGEN_CHECK(n_segs_ >= 1, "PatternChart: the shape has no segments");
        MESHGEN_CHECK((shape.segs.back().to - shape.start).norm() < 1e-6, "PatternChart: a closed shape must end where it starts");

        // 1. flatten in pixels
        std::vector<std::vector<Vec2>> pts; // per kept segment, both end points
        Vec2 cur = shape.start;
        for (int j = 0; j < n_segs_; ++j)
        {
            const PatternSeg& g = shape.segs[j];
            const Vec2 to = j == n_segs_ - 1 ? shape.start : g.to; // close the loop exactly
            std::vector<Vec2> poly = { cur };
            if (g.curve) flatten(cur, g.c1, g.c2, to, flat_tol / scale, poly, 0);
            else poly.push_back(to);
            Real len = 0;
            for (size_t i = 1; i < poly.size(); ++i) len += (poly[i] - poly[i - 1]).norm();
            if (len > 1e-9) { pts.push_back(std::move(poly)); ids_.push_back(j); }
            cur = to;
        }
        MESHGEN_CHECK(pts.size() >= 3, "PatternChart: fewer than 3 non-degenerate segments");

        // 2. metres, y up, centred
        Vec2 lo = Vec2::Constant(1e300), hi = Vec2::Constant(-1e300);
        for (const auto& poly : pts) for (const Vec2& q : poly) { lo = lo.cwiseMin(q); hi = hi.cwiseMax(q); }
        const Vec2 c = 0.5 * (lo + hi);
        for (auto& poly : pts) for (Vec2& q : poly) q = scale * Vec2(q.x() - c.x(), -(q.y() - c.y()));
        for (const auto& poly : pts)
        {
            std::vector<Real> cum(poly.size(), 0.0);
            for (size_t i = 1; i < poly.size(); ++i) cum[i] = cum[i - 1] + (poly[i] - poly[i - 1]).norm();
            cum_.push_back(std::move(cum));
            for (size_t i = 0; i + 1 < poly.size(); ++i)
                if ((poly[i + 1] - poly[i]).norm() > 1e-12) edges_.push_back({ poly[i], poly[i + 1] }); // a segment's end is the next one's start
        }
        pts_ = std::move(pts);

        // 3. polygon area, bounds, simplicity, y-bucket index for contains()
        Real a2 = 0;
        lo_ = Vec2::Constant(1e300); hi_ = Vec2::Constant(-1e300);
        for (const Edge& e : edges_)
        {
            a2 += e.a.x() * e.b.y() - e.b.x() * e.a.y();
            lo_ = lo_.cwiseMin(e.a); hi_ = hi_.cwiseMax(e.a);
        }
        area_ = 0.5 * std::abs(a2);
        MESHGEN_CHECK(area_ > 1e-12, "PatternChart: the contour encloses no area");
        const size_t n = edges_.size();
        for (size_t i = 0; i < n; ++i) // shortcut: O(n^2), a few hundred edges at flat_tol = h/50, bucket the pairs if it ever shows in a profile
            for (size_t k = i + 2; k < n; ++k)
            {
                if (i == 0 && k == n - 1) continue; // first and last edge share the start point
                MESHGEN_CHECK(!intersects(edges_[i], edges_[k]), "PatternChart: the contour crosses itself");
            }
        nb_ = std::max<int>(1, (int)(n / 4));
        buckets_.assign(nb_, {});
        for (size_t i = 0; i < n; ++i)
        {
            const int b0 = bucket(std::min(edges_[i].a.y(), edges_[i].b.y())), b1 = bucket(std::max(edges_[i].a.y(), edges_[i].b.y()));
            for (int b = b0; b <= b1; ++b) buckets_[b].push_back((int)i);
        }
    }

    // Crossing-number parity of a +x ray. Points within 1e-9 of an edge are never decided here in a way that matters: the boundary seeds
    // dominate near the contour (Poisson distance test) and triangle centroids are never on an edge.
    bool contains(const Vec2& p) const override
    {
        if (p.x() < lo_.x() || p.x() > hi_.x() || p.y() < lo_.y() || p.y() > hi_.y()) return false;
        bool in = false;
        for (int i : buckets_[bucket(p.y())])
        {
            const Edge& e = edges_[i];
            if ((e.a.y() > p.y()) != (e.b.y() > p.y()) && p.x() < e.a.x() + (p.y() - e.a.y()) * (e.b.x() - e.a.x()) / (e.b.y() - e.a.y()))
                in = !in;
        }
        return in;
    }

    void bounds(Vec2& lo, Vec2& hi) const override { lo = lo_; hi = hi_; }

    std::vector<BoundaryCurve> sample_boundary(Real h) const override
    {
        std::vector<BoundaryCurve> out;
        for (size_t j = 0; j < pts_.size(); ++j)
        {
            const std::vector<Vec2>& poly = pts_[j];
            const std::vector<Real>& cum = cum_[j];
            const Real L = cum.back();
            const int n = std::max(1, (int)std::lround(L / h));
            BoundaryCurve c{ ids_[j], false, {} };
            c.p.push_back(poly.front());
            for (int k = 1; k < n; ++k)
            {
                const Real s = L * Real(k) / Real(n);
                const size_t i = std::min((size_t)(std::upper_bound(cum.begin(), cum.end(), s) - cum.begin()), cum.size() - 1);
                const Real t = (s - cum[i - 1]) / (cum[i] - cum[i - 1]);
                c.p.push_back(poly[i - 1] + t * (poly[i] - poly[i - 1]));
            }
            c.p.push_back(poly.back()); // both ends: neighbours' shared corners merge in generate()
            out.push_back(std::move(c));
        }
        return out;
    }

    Vec3 to3d(const Vec2& p) const override { return Vec3(p.x(), p.y(), 0.0); }
    Vec3 outward(const Vec3&) const override { return Vec3::UnitZ(); }
    Real area() const override { return area_; }
    int  euler_characteristic() const override { return 1; }
    int  n_boundary_curves() const override { return n_segs_; }

private:
    struct Edge { Vec2 a, b; };

    // Appends the polyline of the cubic (a, c1, c2, b) to `out`, excluding `a`.
    static void flatten(const Vec2& a, const Vec2& c1, const Vec2& c2, const Vec2& b, Real tol, std::vector<Vec2>& out, int depth)
    {
        const Vec2 d = b - a;
        const Real L = d.norm();
        auto dist = [&](const Vec2& q) { return L > 1e-12 ? std::abs(d.x() * (q - a).y() - d.y() * (q - a).x()) / L : (q - a).norm(); };
        if (depth >= 16 || std::max(dist(c1), dist(c2)) <= tol) { out.push_back(b); return; }
        const Vec2 ab = 0.5 * (a + c1), bc = 0.5 * (c1 + c2), cd = 0.5 * (c2 + b), abc = 0.5 * (ab + bc), bcd = 0.5 * (bc + cd), m = 0.5 * (abc + bcd);
        flatten(a, ab, abc, m, tol, out, depth + 1);
        flatten(m, bcd, cd, b, tol, out, depth + 1);
    }

    static bool intersects(const Edge& e, const Edge& f) // closed segments, touching and collinear overlap count
    {
        auto orient = [](const Vec2& a, const Vec2& b, const Vec2& c) { return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x()); };
        auto within = [](const Vec2& a, const Vec2& b, const Vec2& c) // c on the line ab: inside its box?
        { return std::min(a.x(), b.x()) <= c.x() && c.x() <= std::max(a.x(), b.x()) && std::min(a.y(), b.y()) <= c.y() && c.y() <= std::max(a.y(), b.y()); };
        const Real d1 = orient(e.a, e.b, f.a), d2 = orient(e.a, e.b, f.b), d3 = orient(f.a, f.b, e.a), d4 = orient(f.a, f.b, e.b);
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
        return (d1 == 0 && within(e.a, e.b, f.a)) || (d2 == 0 && within(e.a, e.b, f.b)) || (d3 == 0 && within(f.a, f.b, e.a)) || (d4 == 0 && within(f.a, f.b, e.b));
    }

    int bucket(Real y) const
    {
        return std::clamp((int)std::floor((y - lo_.y()) / std::max(hi_.y() - lo_.y(), 1e-300) * nb_), 0, nb_ - 1);
    }

    int n_segs_;
    std::vector<int> ids_;                    // input index of each kept segment
    std::vector<std::vector<Vec2>> pts_;      // per kept segment: flattened polyline (metres), both end points
    std::vector<std::vector<Real>> cum_;      // per kept segment: cumulative arc length
    std::vector<Edge> edges_;                 // the whole polygon, in order
    std::vector<std::vector<int>> buckets_;   // y buckets -> edge indices
    int nb_ = 1;
    Vec2 lo_, hi_;
    Real area_ = 0;
};

// ---- core ---------------------------------------------------------------------------------------------------------

namespace detail {

using Tri = std::array<Index, 3>;

struct Rng // raw mt19937_64 bits -> [0,1): std::uniform_*_distribution is implementation-defined, this is not
{
    std::mt19937_64 gen;
    explicit Rng(uint64_t seed) : gen(seed) {}
    Real next() { return Real(gen() >> 11) * 0x1.0p-53; }
};

inline Real signed_area2(const Vec2& a, const Vec2& b, const Vec2& c)
{
    return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

// A triangulation of the points plus their seam ghosts: corner v of a triangle is the point src[v] shifted by k[v] periods
// (Q[v] holds the shifted coordinates, so every triangle is a plain triangle in one frame). Non-periodic: src = identity, k = 0.
struct Tris { std::vector<Vec2> Q; std::vector<Index> src; std::vector<int> k; std::vector<Tri> tri; };

// Delaunay of the points and of the ghosts of those within 3h of a seam line, centroid-culled against Omega (the chart can be
// non-convex, and each triangle must be kept once), wound CCW in the chart plane.
inline Tris triangulate(const Chart& chart, const std::vector<Vec2>& P, Real h)
{
    Tris out;
    out.Q = P;
    for (Index i = 0; i < (Index)P.size(); ++i) { out.src.push_back(i); out.k.push_back(0); }
    if (chart.periodic())
        for (Index i = 0; i < (Index)P.size(); ++i)
        {
            Real to_lo, to_hi;
            chart.seam_gap(P[i], to_lo, to_hi);
            if (to_lo < 3 * h) { out.Q.push_back(chart.shift(P[i], +1)); out.src.push_back(i); out.k.push_back(+1); }
            if (to_hi < 3 * h) { out.Q.push_back(chart.shift(P[i], -1)); out.src.push_back(i); out.k.push_back(-1); }
        }

    std::vector<double> coords;
    coords.reserve(2 * out.Q.size());
    for (const Vec2& p : out.Q) { coords.push_back(p.x()); coords.push_back(p.y()); }
    try
    {
        delaunator::Delaunator d(coords);
        for (size_t t = 0; t + 2 < d.triangles.size(); t += 3)
        {
            Tri tri = { (Index)d.triangles[t], (Index)d.triangles[t + 1], (Index)d.triangles[t + 2] };
            const Real a2 = signed_area2(out.Q[tri[0]], out.Q[tri[1]], out.Q[tri[2]]);
            if (std::abs(a2) < 1e-12 * h * h) continue;
            if (a2 < 0) std::swap(tri[1], tri[2]);
            if (chart.contains((out.Q[tri[0]] + out.Q[tri[1]] + out.Q[tri[2]]) / 3.0)) out.tri.push_back(tri);
        }
    }
    catch (const std::exception& e)
    {
        MESHGEN_CHECK(false, "Delaunay failed (" << e.what() << "); lower h");
    }
    return out;
}

// Bridson Poisson-disk fill, boundary points are the seeds (respected, not returned). Candidates are wrapped into Omega and
// distances are taken modulo the seam. Candidate order and acceptance are fixed by Rng, so the result depends only on
// (chart, params).
inline std::vector<Vec2> bridson(const Chart& chart, const std::vector<Vec2>& seeds, Real d, int k, Rng& rng)
{
    Vec2 lo, hi;
    chart.bounds(lo, hi);
    lo -= Vec2::Constant(2 * d); // seam images of points within d of a seam land just outside Omega
    hi += Vec2::Constant(2 * d);
    const Real cell = d / std::sqrt(2.0);
    const int nx = (int)std::ceil((hi.x() - lo.x()) / cell) + 1, ny = (int)std::ceil((hi.y() - lo.y()) / cell) + 1;
    MESHGEN_CHECK((Index)nx * ny <= 4000000, "Poisson grid too large; raise h");

    std::vector<std::vector<int>> grid((size_t)nx * ny); // a cell may hold several boundary seeds when d > h
    std::vector<Vec2> pts = seeds;
    auto cell_of = [&](const Vec2& p, int& cx, int& cy)
    {
        cx = std::clamp((int)std::floor((p.x() - lo.x()) / cell), 0, nx - 1);
        cy = std::clamp((int)std::floor((p.y() - lo.y()) / cell), 0, ny - 1);
    };
    auto insert = [&](int i) { int cx, cy; cell_of(pts[i], cx, cy); grid[(size_t)cy * nx + cx].push_back(i); };
    auto near_stored = [&](const Vec2& q)
    {
        int cx, cy;
        cell_of(q, cx, cy);
        for (int y = std::max(cy - 2, 0); y <= std::min(cy + 2, ny - 1); ++y)
            for (int x = std::max(cx - 2, 0); x <= std::min(cx + 2, nx - 1); ++x)
                for (int j : grid[(size_t)y * nx + x])
                    if ((pts[j] - q).squaredNorm() < d * d) return true;
        return false;
    };
    auto is_free = [&](const Vec2& q)
    {
        if (near_stored(q)) return false;
        return !chart.periodic() || !(near_stored(chart.shift(q, +1)) || near_stored(chart.shift(q, -1)));
    };

    std::vector<int> active;
    for (int i = 0; i < (int)pts.size(); ++i) { insert(i); active.push_back(i); }

    while (!active.empty())
    {
        const size_t a = std::min((size_t)(rng.next() * active.size()), active.size() - 1);
        const Vec2 c = pts[active[a]];
        bool placed = false;
        for (int t = 0; t < k && !placed; ++t)
        {
            const Real rad = d * (1.0 + rng.next()), ang = kTwoPi * rng.next();
            const Vec2 q = chart.wrap(c + rad * Vec2(std::cos(ang), std::sin(ang))); // leaving through one seam side re-enters on the other
            if (!chart.contains(q) || !is_free(q)) continue;
            pts.push_back(q);
            insert((int)pts.size() - 1);
            active.push_back((int)pts.size() - 1);
            placed = true;
        }
        if (!placed) { active[a] = active.back(); active.pop_back(); }
        MESHGEN_CHECK(pts.size() <= 50000, "more than 50000 points; raise h");
    }
    return std::vector<Vec2>(pts.begin() + (Index)seeds.size(), pts.end());
}

// Move every free point to the area-weighted centroid of its incident triangles (Jacobi: all from the old positions). A ghost
// corner votes for its source point, with the centroid shifted back into the source's frame.
inline void smooth(const Chart& chart, std::vector<Vec2>& P, Index n_fixed, Real h)
{
    const Tris T = triangulate(chart, P, h);
    std::vector<Vec2> num(P.size(), Vec2::Zero());
    std::vector<Real> den(P.size(), 0.0);
    for (const Tri& t : T.tri)
    {
        const Real a = 0.5 * std::abs(signed_area2(T.Q[t[0]], T.Q[t[1]], T.Q[t[2]]));
        const Vec2 c = (T.Q[t[0]] + T.Q[t[1]] + T.Q[t[2]]) / 3.0;
        for (Index v : t) { num[T.src[v]] += a * chart.shift(c, -T.k[v]); den[T.src[v]] += a; }
    }
    for (Index i = n_fixed; i < (Index)P.size(); ++i)
    {
        if (den[i] <= 0) continue;
        const Vec2 q = chart.wrap(num[i] / den[i]);
        if (chart.contains(q)) P[i] = q;
    }
}

} // namespace detail

inline Report validate(const TriSurface& s, const Chart& chart, const Params& params)
{
    Report r;
    r.vertices = (Index)s.pos.size();
    r.triangles = (Index)s.tris.size();
    std::map<std::pair<Index,Index>, int> count;
    std::vector<int> used(s.pos.size(), 0);
    Real area = 0, ang_sum = 0, ang_min = 180;
    for (const auto& t : s.tris)
    {
        area += 0.5 * (s.pos[t[1]] - s.pos[t[0]]).cross(s.pos[t[2]] - s.pos[t[0]]).norm();
        for (int c = 0; c < 3; ++c)
        {
            const Index a = t[c], b = t[(c + 1) % 3];
            used[a] = 1;
            count[{ std::min(a, b), std::max(a, b) }]++;
            const Vec3 u = (s.pos[b] - s.pos[a]).normalized(), v = (s.pos[t[(c + 2) % 3]] - s.pos[a]).normalized();
            const Real ang = std::acos(std::clamp(u.dot(v), -1.0, 1.0)) * 180.0 / (kTwoPi / 2);
            ang_sum += ang; ang_min = std::min(ang_min, ang);
        }
    }
    r.edges = (Index)count.size();
    r.euler = (int)(r.vertices - r.edges + r.triangles);
    r.edge_min = 1e300;
    for (const auto& [e, c] : count)
    {
        const Real len = (s.pos[e.first] - s.pos[e.second]).norm() / params.h;
        r.edge_min = std::min(r.edge_min, len); r.edge_max = std::max(r.edge_max, len); r.edge_mean += len;
        if (c > 2) ++r.nonmanifold;
        if (c == 1)
        {
            ++r.open_edges;
            const int nb = chart.n_boundary_curves(); // a legal free edge joins two vertices of free-edge curves
            if (!(s.curve[e.first] >= 0 && s.curve[e.first] < nb && s.curve[e.second] >= 0 && s.curve[e.second] < nb)) ++r.bad_open_edges;
        }
    }
    if (r.edges) r.edge_mean /= Real(r.edges);
    for (int u : used) r.unused_vertices += !u;
    r.min_angle_deg = ang_min;
    r.mean_angle_deg = r.triangles ? ang_sum / Real(3 * r.triangles) : 0;
    r.area_error = area / chart.area() - 1.0;
    for (size_t i = 0; i < s.uv.size(); ++i)
    {
        Real to_lo, to_hi;
        chart.seam_gap(s.uv[i], to_lo, to_hi);
        r.interior_on_seam += s.curve[i] < 0 && std::min(to_lo, to_hi) < 1e-9;
    }
    r.structure_ok = r.triangles > 0 && r.euler == chart.euler_characteristic() && r.nonmanifold == 0
                  && r.bad_open_edges == 0 && r.unused_vertices == 0;
    r.quality_ok = std::abs(r.area_error) < 0.03 && r.min_angle_deg > 15.0 && r.edge_min > 0.4 && r.interior_on_seam == 0;
    return r;
}

inline TriSurface generate(const Chart& chart, const Params& params)
{
    MESHGEN_CHECK(params.h > 0 && params.beta > 0 && params.beta <= 1.0, "need h > 0 and 0 < beta <= 1");
    const Real h = params.h;

    // 1. boundary, de-duplicated across curves (first writer wins: custom shapes share corners); `cidx[c][j]` = index into P
    const std::vector<BoundaryCurve> curves = chart.sample_boundary(h);
    Vec2 lo, hi;
    chart.bounds(lo, hi);
    const Real merge_dist = 1e-9 * (hi - lo).norm();
    std::vector<Vec2> P;
    std::vector<int> curve;
    std::vector<std::vector<Index>> cidx;
    for (const BoundaryCurve& c : curves)
    {
        cidx.emplace_back();
        for (const Vec2& p : c.p)
        {
            Index found = -1;
            for (Index j = 0; j < (Index)P.size() && found < 0; ++j) // shortcut: O(n^2), boundary only, upgrade if it ever shows in a profile
                if ((P[j] - p).norm() < merge_dist) found = j;
            if (found < 0) { found = (Index)P.size(); P.push_back(p); curve.push_back(c.id); }
            cidx.back().push_back(found);
        }
    }
    const Index n_fixed = (Index)P.size();

    // 2. Poisson-disk interior, 3. smoothing, 4. final triangulation
    detail::Rng rng(params.seed);
    const std::vector<Vec2> inner = detail::bridson(chart, P, params.beta * h, params.k, rng);
    P.insert(P.end(), inner.begin(), inner.end());
    curve.resize(P.size(), -1);
    for (int it = 0; it < params.smooth_iters; ++it) detail::smooth(chart, P, n_fixed, h);
    const detail::Tris T = detail::triangulate(chart, P, h);

    // 5. the mesh is on the source points: dropping a corner's ghost index is the weld
    TriSurface s;
    s.uv = P;
    s.curve = curve;
    s.boundary = cidx;
    for (const Vec2& p : P) s.pos.push_back(chart.to3d(p));
    std::set<std::pair<Index,Index>> flat_edges;
    for (const auto& t : T.tri)
    {
        const std::array<Index, 3> w = { T.src[t[0]], T.src[t[1]], T.src[t[2]] };
        MESHGEN_CHECK(w[0] != w[1] && w[1] != w[2] && w[0] != w[2], "triangle with a repeated vertex (surface narrower than a few edges); lower h");
        s.tris.push_back({ w[0], w[1], w[2] });
        for (int c = 0; c < 3; ++c) flat_edges.insert({ std::min(w[c], w[(c + 1) % 3]), std::max(w[c], w[(c + 1) % 3]) });
    }
    // unconstrained Delaunay need not contain every boundary segment
    for (size_t c = 0; c < curves.size(); ++c)
    {
        const size_t n = cidx[c].size();
        for (size_t j = 0; j < (curves[c].closed ? n : n - 1); ++j)
        {
            const Index a = cidx[c][j], b = cidx[c][(j + 1) % n];
            MESHGEN_CHECK(flat_edges.count({ std::min(a, b), std::max(a, b) }), "boundary segment missing from the triangulation; lower h");
        }
    }

    // 6. orientation: tris are CCW in the chart; flip all if that points against the chart's outward side
    {
        const auto& t = s.tris.front();
        const Vec3 n = (s.pos[t[1]] - s.pos[t[0]]).cross(s.pos[t[2]] - s.pos[t[0]]);
        const Vec3 c = (s.pos[t[0]] + s.pos[t[1]] + s.pos[t[2]]) / 3.0;
        if (n.dot(chart.outward(c)) < 0) for (auto& tri : s.tris) std::swap(tri[1], tri[2]);
    }

    // 7. edges and lumped areas
    std::set<std::pair<Index,Index>> edge_set;
    s.vertex_area.assign(s.pos.size(), 0.0);
    for (const auto& t : s.tris)
    {
        const Real a = 0.5 * (s.pos[t[1]] - s.pos[t[0]]).cross(s.pos[t[2]] - s.pos[t[0]]).norm();
        for (int c = 0; c < 3; ++c)
        {
            s.vertex_area[t[c]] += a / 3.0;
            edge_set.insert({ std::min(t[c], t[(c + 1) % 3]), std::max(t[c], t[(c + 1) % 3]) });
        }
    }
    s.edges.assign(edge_set.begin(), edge_set.end());

    const Report r = validate(s, chart, params);
    MESHGEN_CHECK(r.structure_ok, "mesh is not a clean surface (euler " << r.euler << ", open edges off the boundary " << r.bad_open_edges
           << ", nonmanifold edges " << r.nonmanifold << ", unused vertices " << r.unused_vertices << "); lower h");
    return s;
}

} // namespace meshgen
