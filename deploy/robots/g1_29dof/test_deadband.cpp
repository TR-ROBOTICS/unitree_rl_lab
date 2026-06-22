// Standalone unit test for the pressure-deadband hysteresis logic.
// No DDS, no ONNX, no ROS — just drives the latch state machine with injected
// p_now/p_des values and verifies ENTER/EXIT transitions.
//
// Build & run (no cmake needed):
//   g++ -std=c++17 -o /tmp/test_deadband \
//       /home/jescobars/unitree_rl_lab/deploy/robots/g1_29dof/test_deadband.cpp \
//       && /tmp/test_deadband
//
// ADR 0012 / sim2sim deadband — unit test.

#include <cmath>
#include <cstdio>
#include <cstdlib>

// Inline the exact same logic as State_ValveTurn's policy thread loop.
// If the logic changes there, update here too.
static int failures = 0;

static void check(const char* label, bool cond)
{
    if (cond) {
        std::fprintf(stdout, "  PASS  %s\n", label);
    } else {
        std::fprintf(stderr, "  FAIL  %s\n", label);
        ++failures;
    }
}

struct DeadbandSM {
    float epsilon_enter;
    float epsilon_exit;
    bool  in_hold = false;

    // Returns true if in_hold changed this tick (for logging).
    bool step(float p_now, float p_des)
    {
        float err = std::fabs(p_des - p_now);
        bool prev = in_hold;

        if (!in_hold && err < epsilon_enter) {
            in_hold = true;
            std::fprintf(stdout, "    ENTER HOLD  err=%.2f p_now=%.1f p_des=%.1f thr=%.1f\n",
                         err, p_now, p_des, epsilon_enter);
        } else if (in_hold && err > epsilon_exit) {
            in_hold = false;
            std::fprintf(stdout, "    EXIT  HOLD  err=%.2f p_now=%.1f p_des=%.1f thr=%.1f\n",
                         err, p_now, p_des, epsilon_exit);
        }
        return in_hold != prev;
    }
};

int main()
{
    const float EPS_ENTER = 2.0f;
    const float EPS_EXIT  = 4.0f;
    DeadbandSM sm{EPS_ENTER, EPS_EXIT};

    std::fprintf(stdout, "=== deadband unit test  enter=%.1f exit=%.1f ===\n",
                 EPS_ENTER, EPS_EXIT);

    // -----------------------------------------------------------------------
    // 1. Far from target: no hold.
    std::fprintf(stdout, "\n[1] err=20 PSI — should stay OUT of hold\n");
    sm.step(80.0f, 100.0f);   // err = 20
    check("not in_hold at err=20", !sm.in_hold);

    // -----------------------------------------------------------------------
    // 2. Cross enter threshold from above (err drops below 2).
    std::fprintf(stdout, "\n[2] err=1 PSI — should ENTER hold\n");
    sm.step(99.0f, 100.0f);   // err = 1  < 2
    check("in_hold after err=1", sm.in_hold);

    // -----------------------------------------------------------------------
    // 3. Chatter zone: err stays between 2 and 4 — hold must NOT exit.
    std::fprintf(stdout, "\n[3] err=3 PSI — in hysteresis band, should stay IN hold\n");
    sm.step(97.0f, 100.0f);   // err = 3  (enter=2, exit=4) → no change
    check("still in_hold at err=3 (no chatter)", sm.in_hold);

    // -----------------------------------------------------------------------
    // 4. Back inside band (even tighter) — still no exit.
    std::fprintf(stdout, "\n[4] err=0.5 PSI — tighter, stays IN hold\n");
    sm.step(99.5f, 100.0f);
    check("still in_hold at err=0.5", sm.in_hold);

    // -----------------------------------------------------------------------
    // 5. Cross exit threshold: err > 4 — should EXIT hold.
    std::fprintf(stdout, "\n[5] err=5 PSI — should EXIT hold\n");
    sm.step(95.0f, 100.0f);   // err = 5  > 4
    check("not in_hold after err=5", !sm.in_hold);

    // -----------------------------------------------------------------------
    // 6. Enter again immediately (err < 2).
    std::fprintf(stdout, "\n[6] re-enter: err=1.5 PSI\n");
    sm.step(98.5f, 100.0f);
    check("in_hold after re-enter", sm.in_hold);

    // -----------------------------------------------------------------------
    // 7. Exactly at epsilon_enter (not strictly less) — must NOT enter a fresh hold.
    //    Reset first.
    sm.in_hold = false;
    std::fprintf(stdout, "\n[7] err == epsilon_enter (2.0) exactly — should NOT enter\n");
    sm.step(98.0f, 100.0f);   // err = 2.0, condition is strictly <
    check("not in_hold at err==epsilon_enter", !sm.in_hold);

    // -----------------------------------------------------------------------
    // 8. Exactly at epsilon_exit while in hold — must NOT exit (strictly >).
    sm.in_hold = true;
    std::fprintf(stdout, "\n[8] err == epsilon_exit (4.0) while in hold — should NOT exit\n");
    sm.step(96.0f, 100.0f);   // err = 4.0, condition is strictly >
    check("still in_hold at err==epsilon_exit", sm.in_hold);

    // -----------------------------------------------------------------------
    std::fprintf(stdout, "\n=== result: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
