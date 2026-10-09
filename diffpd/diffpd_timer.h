#pragma once

#include "diffpd_types.h"

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

// Named stopwatch accumulator: TIMER_START("name") ... TIMER_END("name") adds the elapsed time to that
// stage (repeated calls accumulate). Stages stay in order of first use. Does nothing unless
// `stage_timer.enabled`, so only the run that sets it is measured (e.g. not the FD-check re-simulations);
// build with -DDIFFPD_TIMERS=0 to compile the calls out entirely.
#ifndef DIFFPD_TIMERS
#define DIFFPD_TIMERS 1
#endif

// One row of a timing breakdown: stage name and its fraction (0..1) of the total.
struct StageShare { std::string name; double fraction; };

struct StageTimer
{
    using Clock = std::chrono::steady_clock;
    struct Stage { std::string name; double seconds = 0.0; Clock::time_point start; bool running = false; };

    bool               enabled = false;
    std::vector<Stage> stages;

    void reset() { stages.clear(); }

    void start(std::string_view name)
    {
        if (!enabled) return;
        Stage& s = find_or_add(name);
        ASSERT(!s.running, "TIMER_START(\"" << name << "\") while already running");
        s.running = true;
        s.start   = Clock::now(); // last, so the bookkeeping above isn't measured
    }

    void end(std::string_view name)
    {
        if (!enabled) return;
        const Clock::time_point now = Clock::now();
        Stage& s = find_or_add(name);
        ASSERT(s.running, "TIMER_END(\"" << name << "\") without a matching TIMER_START");
        s.running  = false;
        s.seconds += std::chrono::duration<double>(now - s.start).count();
    }

    // Each stage's fraction of the total, where total = `total_stage` minus `excluded_stage` (e.g. whole
    // loop minus viewer drawing). Whatever the stages don't cover is appended as "other".
    std::vector<StageShare> shares(std::string_view total_stage, std::string_view excluded_stage) const
    {
        const double total = seconds_of(total_stage) - seconds_of(excluded_stage);
        std::vector<StageShare> out;
        double covered = 0.0;
        for (const Stage& s : stages)
        {
            if (s.name == total_stage || s.name == excluded_stage) continue;
            out.push_back({ s.name, s.seconds / total });
            covered += s.seconds;
        }
        out.push_back({ "other", (total - covered) / total });
        return out;
    }

private:
    Stage& find_or_add(std::string_view name)
    {
        for (Stage& s : stages)
            if (s.name == name) return s;
        stages.push_back(Stage{std::string(name)});
        return stages.back();
    }

    double seconds_of(std::string_view name) const
    {
        for (const Stage& s : stages)
            if (s.name == name) return s.seconds;
        return 0.0;
    }
};

inline StageTimer stage_timer;

#if DIFFPD_TIMERS
#define TIMER_START(name) stage_timer.start(name)
#define TIMER_END(name)   stage_timer.end(name)
#else
#define TIMER_START(name) ((void)0)
#define TIMER_END(name)   ((void)0)
#endif
