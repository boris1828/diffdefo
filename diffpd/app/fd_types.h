#pragma once

// What the gradient / finite-difference check reports and the viewer displays. Plain data, no solver code.

#include "core/diffpd_core.h"

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

// FD-check epsilon choices, indexed like AppConfig::fd_eps_selected (0 = 1e-2 ... 8 = 1e-10).
inline constexpr Real kFDEpsilonValues[9] = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };

// Loss/gradient summary for one guess/target pair, for playback-screen display. dphi_dx0/dphi_dv0
// are collapsed from per-particle vectors to their column sums (net gradient per world axis).
struct GradientSummary
{
    Real loss     = 0.0;
    Vec3 dphi_dx0 = Vec3::Zero();
    Vec3 dphi_dv0 = Vec3::Zero();
    // {"dphi/dk", g} when stiffness is unified, else one {"dphi/dk_<kind>", g} per enabled spring
    // kind, stretch first. front() is the gradient the FD check verifies.
    std::vector<std::pair<std::string, Real>> dphi_dk;
};

// Result of one central-difference check of dphi/dk against the analytic gradient; shared here so
// the viewer can display it without knowing about Loss/pd_contact/Object construction.
struct FDCheckResult
{
    Real fd;
    Real analytic;
    Real rel_err;
};

// FD results of the current experiment, indexed like kFDEpsilonValues; an empty slot = not checked yet.
using FDCheckCache = std::array<std::optional<FDCheckResult>, 9>;

// Runs a stiffness FD check for every selected epsilon (indexed like kFDEpsilonValues) that isn't
// already in the experiment's FDCheckCache, storing results there. diffpd.cpp builds the actual
// closure (and owns the cache); the viewer just calls it with its checkbox state.
using FDCheckRunner = std::function<void(const bool (&selected)[9])>;
