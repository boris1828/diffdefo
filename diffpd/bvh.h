#pragma once

// Triangle-mesh geometry queries, GL-free: closest point on a triangle (with barycentrics) and a
// bounding-volume hierarchy over a triangle soup (build, refit for a deforming mesh, closest-point query).
// The data types (BvhBox, BvhNode, Bvh, ContactFeature) live in diffpd_types.h.

#include "diffpd_types.h"

#include <limits>
#include <numeric>

// Closest point to p on triangle (a, b, c) (Ericson, Real-Time Collision Detection 5.1.5), plus the
// region it lands in and its barycentric coordinates (weights of a, b, c; sum to 1). Edge e runs from
// vertex e to (e+1)%3, matching TriMesh::edge_pseudo_normal.
inline Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c,
                                      ContactFeature& feature, int& index, Vec3& bary)
{
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const Real d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) { feature = ContactFeature::Vertex; index = 0; bary = Vec3(1, 0, 0); return a; }

    const Vec3 bp = p - b;
    const Real d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) { feature = ContactFeature::Vertex; index = 1; bary = Vec3(0, 1, 0); return b; }

    const Real vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
    {
        const Real s = d1 / (d1 - d3);
        feature = ContactFeature::Edge; index = 0; bary = Vec3(1 - s, s, 0);
        return a + s * ab;
    }

    const Vec3 cp = p - c;
    const Real d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) { feature = ContactFeature::Vertex; index = 2; bary = Vec3(0, 0, 1); return c; }

    const Real vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
    {
        const Real s = d2 / (d2 - d6);
        feature = ContactFeature::Edge; index = 2; bary = Vec3(1 - s, 0, s);
        return a + s * ac;
    }

    const Real va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    {
        const Real s = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        feature = ContactFeature::Edge; index = 1; bary = Vec3(0, 1 - s, s);
        return b + s * (c - b);
    }

    const Real denom = 1.0 / (va + vb + vc);
    const Real v = vb * denom, w = vc * denom;
    feature = ContactFeature::Face; index = 0; bary = Vec3(1 - v - w, v, w);
    return a + ab * v + ac * w;
}

inline Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c,
                                      ContactFeature& feature, int& index)
{
    Vec3 bary;
    return closest_point_on_triangle(p, a, b, c, feature, index, bary);
}

// Purely geometric result of a closest-point query (no inside/outside sign: see trimesh.h).
struct ClosestHit
{
    Vec3           point = Vec3::Zero(); // closest point on the surface
    Vec3           bary  = Vec3::Zero(); // its barycentric coordinates in `triangle`
    Real           dist2 = std::numeric_limits<Real>::infinity();
    int            triangle = -1;        // -1 = nothing within max_dist2
    ContactFeature feature  = ContactFeature::Face;
    int            index    = 0;         // edge/vertex slot within `triangle`
};

namespace bvh_detail
{
inline BvhBox triangle_box(const std::vector<Vec3>& V, const std::array<int, 3>& t)
{
    BvhBox b;
    b.lo = V[t[0]].cwiseMin(V[t[1]]).cwiseMin(V[t[2]]);
    b.hi = V[t[0]].cwiseMax(V[t[1]]).cwiseMax(V[t[2]]);
    return b;
}

inline BvhBox merge(const BvhBox& a, const BvhBox& b) { return BvhBox{ a.lo.cwiseMin(b.lo), a.hi.cwiseMax(b.hi) }; }

// Squared distance from p to the box (0 if inside).
inline Real box_distance2(const BvhBox& b, const Vec3& p)
{
    const Vec3 d = (b.lo - p).cwiseMax(p - b.hi).cwiseMax(Vec3::Zero());
    return d.squaredNorm();
}

inline BvhBox leaf_box(const Bvh& bvh, const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T,
                       int first, int count)
{
    BvhBox box = triangle_box(V, T[bvh.tri_ids[first]]);
    for (int k = 1; k < count; ++k) box = merge(box, triangle_box(V, T[bvh.tri_ids[first + k]]));
    return box;
}

// Builds the subtree over tri_ids[first, first + count) and returns its node index. The node is pushed
// before its children, which keeps child indices above their parent's.
inline int build_node(Bvh& bvh, const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T,
                      const std::vector<Vec3>& centroid, int first, int count, int leaf_size)
{
    const int id = (int)bvh.nodes.size();
    bvh.nodes.emplace_back();
    const BvhBox box = leaf_box(bvh, V, T, first, count);
    bvh.nodes[id].box = box;

    if (count <= leaf_size)
    {
        bvh.nodes[id].first = first;
        bvh.nodes[id].count = count;
        return id;
    }

    // Median split on the longest axis of the triangle centroids.
    Vec3 clo = centroid[bvh.tri_ids[first]], chi = clo;
    for (int k = 1; k < count; ++k)
    {
        clo = clo.cwiseMin(centroid[bvh.tri_ids[first + k]]);
        chi = chi.cwiseMax(centroid[bvh.tri_ids[first + k]]);
    }
    int axis = 0;
    (chi - clo).maxCoeff(&axis);

    const int mid = count / 2;
    std::nth_element(bvh.tri_ids.begin() + first, bvh.tri_ids.begin() + first + mid, bvh.tri_ids.begin() + first + count,
                     [&](int a, int b) { return centroid[a][axis] < centroid[b][axis]; });

    const int left  = build_node(bvh, V, T, centroid, first, mid, leaf_size);
    const int right = build_node(bvh, V, T, centroid, first + mid, count - mid, leaf_size);
    bvh.nodes[id].left  = left;
    bvh.nodes[id].right = right;
    return id;
}
} // namespace bvh_detail

// Builds the hierarchy over triangles T of vertices V. Triangle ids are indices into T.
inline void bvh_build(Bvh& bvh, const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T, int leaf_size = 4)
{
    bvh.nodes.clear();
    bvh.tri_ids.resize(T.size());
    std::iota(bvh.tri_ids.begin(), bvh.tri_ids.end(), 0);
    if (T.empty()) return;

    std::vector<Vec3> centroid(T.size());
    for (size_t t = 0; t < T.size(); ++t) centroid[t] = (V[T[t][0]] + V[T[t][1]] + V[T[t][2]]) / 3.0;

    bvh.nodes.reserve(2 * T.size() / std::max(1, leaf_size) + 1);
    bvh_detail::build_node(bvh, V, T, centroid, 0, (int)T.size(), leaf_size);
}

// Recomputes every node box for moved vertices (same triangles, same tree): leaves from their triangles,
// inner nodes as the union of their children. One pass in reverse order, since children follow parents.
inline void bvh_refit(Bvh& bvh, const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T)
{
    for (int i = (int)bvh.nodes.size() - 1; i >= 0; --i)
    {
        BvhNode& n = bvh.nodes[i];
        n.box = n.count > 0 ? bvh_detail::leaf_box(bvh, V, T, n.first, n.count)
                            : bvh_detail::merge(bvh.nodes[n.left].box, bvh.nodes[n.right].box);
    }
}

// Closest point on the mesh to p: nearer-child-first traversal pruned by box distance. Only surface
// points strictly closer than sqrt(max_dist2) count; if there are none, `triangle` stays -1. Ties between
// triangles sharing the closest point (an edge or a vertex) may resolve either way — the point, distance
// and feature are the same.
inline ClosestHit bvh_closest(const Bvh& bvh, const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T,
                              const Vec3& p, Real max_dist2 = std::numeric_limits<Real>::infinity())
{
    ClosestHit best;
    best.dist2 = max_dist2;
    if (bvh.nodes.empty()) return best;

    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top > 0)
    {
        const BvhNode& node = bvh.nodes[stack[--top]];

        if (node.count > 0)
        {
            for (int k = 0; k < node.count; ++k)
            {
                const int                  t   = bvh.tri_ids[node.first + k];
                const std::array<int, 3>&  tri = T[t];
                ContactFeature feature;
                int            index;
                Vec3           bary;
                const Vec3     q  = closest_point_on_triangle(p, V[tri[0]], V[tri[1]], V[tri[2]], feature, index, bary);
                const Real     d2 = (p - q).squaredNorm();
                if (d2 < best.dist2)
                {
                    best.dist2 = d2; best.point = q; best.bary = bary;
                    best.triangle = t; best.feature = feature; best.index = index;
                }
            }
            continue;
        }

        const Real dl = bvh_detail::box_distance2(bvh.nodes[node.left].box, p);
        const Real dr = bvh_detail::box_distance2(bvh.nodes[node.right].box, p);
        // Push the farther child first so the nearer one is popped (and tightens `best`) first.
        if (dl <= dr)
        {
            if (dr <= best.dist2) stack[top++] = node.right;
            if (dl <= best.dist2) stack[top++] = node.left;
        }
        else
        {
            if (dl <= best.dist2) stack[top++] = node.left;
            if (dr <= best.dist2) stack[top++] = node.right;
        }
    }
    return best;
}
