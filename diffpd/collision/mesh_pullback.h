#pragma once

// Gradient of a mesh collider's contact geometry with respect to the mesh VERTICES (not the particle): the last step of the backward
// pass for an experiment that moves the body (e.g. fits a body model to a cloth). The backward pass hands each active contact the
// adjoints lambda_n of its normal n and lambda_phi of its gap phi (solver/impulse/backward.h); this pulls them back through the
// triangle the contact touched, for the loss l = lambda_n . n + lambda_phi phi. Gap contact only: the geometry is read at x-, where
// stamp_contacts evaluated it. Derivation: CLAUDE.md, "Triangle-mesh colliders"; checked by `diffpd --mesh-vertex-grad-check`.
//
//   gap     every feature   g_k = -lambda_phi bary_k grad(phi)     (bary_k = the weight of corner k at the surface point: the lever rule)
//   normal  face            g_k = (lambda_n x e_k) / (2 area)       (corner k tilts the plane about its opposite edge e_k, counter-clockwise)
//           shell vertex    g_i = -lambda_n / d                     (a sphere of radius d around the vertex)
//           shell edge      g_i = -(1-s) nu_perp + (nu.e/|e|) u,  g_j = -s nu_perp - (nu.e/|e|) u,  nu = lambda_n / d,  u = p - q = d n
//   a shell contact is one outside the surface with thickness > 0 (offset_mesh_contact): Contact::inv_r = 1/d > 0. An edge or vertex
//   contact without it uses a pseudo-normal, which depends on the whole 1-ring of faces: its normal term is skipped (see the return value).

#include "collision/trimesh.h"
#include "collision/world.h"

// Adds dl/d(rest vertices) of the contact `c` to g (one Vec3 per mesh.vertices entry, the mesh's rest frame; world = pose.position + pose.rotation v).
// lambda_n: the tangent part is used (n stays unit). Returns false if the normal term had to be skipped (pseudo-normal edge / vertex contact); the gap
// term is added anyway.
inline bool mesh_vertex_pullback(const TriMesh& mesh, const RigidPose& pose, const Contact& c, const Vec3& lambda_n, Real lambda_phi, std::vector<Vec3>& g)
{
    ASSERT(c.triangle_id >= 0, "mesh_vertex_pullback: the contact has no triangle");
    const std::array<int, 3>& tri = mesh.triangles[c.triangle_id];
    Vec3 P[3], gw[3]; // corners in the world frame, and l's gradient with respect to them
    for (int k = 0; k < 3; ++k)
    {
        P[k]  = pose.position + pose.rotation * mesh.vertices[tri[k]];
        gw[k] = -lambda_phi * c.bary[k] * c.gap_grad;
    }

    const Vec3 lam = lambda_n - c.normal * c.normal.dot(lambda_n);
    bool normal_done = true;
    if (c.feature == ContactFeature::Face)
    {
        const Vec3 e[3] = { P[2] - P[1], P[0] - P[2], P[1] - P[0] }; // the edge opposite corner k
        const Real two_area = (P[1] - P[0]).cross(P[2] - P[0]).norm();
        for (int k = 0; k < 3; ++k) gw[k] += lam.cross(e[k]) / two_area;
    }
    else if (c.inv_r > 0.0) // shell: the normal is radial, n = (p - q) / d
    {
        const Real d  = 1.0 / c.inv_r;
        const Vec3 nu = lam / d;
        if (c.feature == ContactFeature::Vertex) gw[c.feature_index] -= nu;
        else
        {
            const int  i = c.feature_index, j = (i + 1) % 3;
            const Vec3 e = P[j] - P[i];
            const Real el = e.norm(), s = c.bary[j]; // s: the surface point's fraction of the way from corner i to corner j
            const Vec3 eh = e / el, nu_perp = nu - eh * eh.dot(nu), twist = (nu.dot(eh) / el) * (d * c.normal);
            gw[i] += -(1.0 - s) * nu_perp + twist;
            gw[j] += -s * nu_perp - twist;
        }
    }
    else normal_done = false; // shortcut: pseudo-normal edge / vertex contact, its normal term is dropped; use thickness > 0 so none occur

    for (int k = 0; k < 3; ++k) g[tri[k]] += pose.rotation.conjugate() * gw[k];
    return normal_done;
}
