#include "State_ValveTurn.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>

// --- static members ---
std::atomic<float> State_ValveTurn::p_now_psi{15.0f};    // updated from rt/valve/pressure
std::atomic<float> State_ValveTurn::p_des_psi{100.0f};   // overwritten from cfg
constexpr int State_ValveTurn::kArmSdkIdx[14];


namespace isaaclab
{
namespace mdp
{

// p_now sourced from rt/valve/pressure (valve_pressure_node.py applies g(theta)).
REGISTER_OBSERVATION(valve_pressure_now)
{
    constexpr float P_SPAN = 185.0f;
    return std::vector<float>{State_ValveTurn::p_now_psi.load() / P_SPAN};
}

REGISTER_OBSERVATION(valve_pressure_des)
{
    constexpr float P_SPAN = 185.0f;
    return std::vector<float>{State_ValveTurn::p_des_psi.load() / P_SPAN};
}

}  // namespace mdp
}  // namespace isaaclab


State_ValveTurn::State_ValveTurn(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    if (cfg["p_des"]) {
        p_des_psi.store(cfg["p_des"].as<float>());
    }
    spdlog::info("[ValveTurn] target pressure p_des = {:.1f} PSI", p_des_psi.load());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // Subscribe to p_now from valve_pressure_node.py (applies g(theta) centrally).
    p_now_sub_ = std::make_shared<unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Point_>>("rt/valve/pressure");
    p_now_sub_->InitChannel([](const void* message) {
        auto* p = static_cast<const geometry_msgs::msg::dds_::Point_*>(message);
        State_ValveTurn::p_now_psi.store(static_cast<float>(p->x()));
    }, 1);

    // Subscribe to live p_des updates from valve_pressure_node.py.
    // Config yaml sets the initial value; this topic allows runtime changes without restart.
    p_des_sub_ = std::make_shared<unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Point_>>("rt/valve/pressure_des");
    p_des_sub_->InitChannel([](const void* message) {
        constexpr float P_MIN = 15.0f, P_MAX = 200.0f;
        auto* p = static_cast<const geometry_msgs::msg::dds_::Point_*>(message);
        float val = std::clamp(static_cast<float>(p->x()), P_MIN, P_MAX);
        State_ValveTurn::p_des_psi.store(val);
    }, 1);

}

void State_ValveTurn::enter()
{
    // Gains by SDK motor index (deploy.yaml stiffness/damping are SDK-ordered).
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    // Hold every joint at the crane-hang/pregrip pose. default_joint_pos is in
    // deploy order; joint_ids_map maps it to SDK motor indices. run() then
    // overwrites the 14 arm joints each tick; the 15 body joints stay held.
    auto& ids = env->robot->data.joint_ids_map;
    for (int i = 0; i < ids.size(); ++i) {
        lowcmd->msg_.motor_cmd()[ids[i]].q() = env->robot->data.default_joint_pos[i];
    }

    env->robot->update();

    policy_thread_running = true;
    policy_thread = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        env->reset();

        while (policy_thread_running)
        {
            env->step();
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_ValveTurn::run()
{
    // processed_actions() = 14 arm targets in ONNX/Isaac action order.
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < 14; ++i) {
        lowcmd->msg_.motor_cmd()[kArmSdkIdx[i]].q() = action[i];
    }
}
