#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/ros2/Point_.hpp>

#include <atomic>
#include <memory>
#include <thread>

// Valve-turn deploy state. Drives the 14 G1 arm joints from the v5_i policy while
// holding the 15 body joints at the crane-hang pregrip pose. The valve hinge is a
// passive joint in the MuJoCo sim (no actuator) so its angle never reaches the
// controller via motor_state; the bridge publishes it on rt/valve/angle and this
// state subscribes, exposing theta to the valve_pressure_now observation term.
class State_ValveTurn : public FSMState
{
public:
    State_ValveTurn(int state_mode, std::string state_string);

    void enter();
    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    // p_now (PSI) from rt/valve/pressure; p_des (PSI) initially from FSM config,
    // overwritten live via rt/valve/pressure_des. Both updated by DDS subscribers.
    // Static so the registered observation terms can read them.
    static std::atomic<float> p_now_psi;
    static std::atomic<float> p_des_psi;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::shared_ptr<unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Point_>> p_now_sub_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Point_>> p_des_sub_;

    // SDK motor index for each of the 14 policy arm outputs, in ONNX action order.
    // Resolved 2026-06-15 via scripts/dump_valve_deploy.py:
    // Lsp,Rsp,Lsr,Rsr,Lsy,Rsy,Lel,Rel,Lwr_roll,Rwr_roll,Lwp,Rwp,Lwy,Rwy.
    static constexpr int kArmSdkIdx[14] =
        {15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

    std::thread policy_thread;
    bool policy_thread_running = false;

    // ADR 0012 / sim2sim deadband — pressure hysteresis HOLD.
    // Set by the constructor from config.yaml (defaults live there, not here).
    float epsilon_enter_;  // enter HOLD when |err| < epsilon_enter_ (PSI)
    float epsilon_exit_;   // exit  HOLD when |err| > epsilon_exit_  (PSI)
    // Latch: true while pressure is within the hold band.
    // Written and read only from the policy thread — no atomics needed.
    bool in_hold_ = false;
};

REGISTER_FSM(State_ValveTurn)
