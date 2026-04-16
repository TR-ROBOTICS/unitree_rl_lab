#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

// Arm pose command observation
// LB + up   → lift left arm  (left_target  = 1.57 rad)
// LB + down → lower left arm (left_target  = 0.0  rad)
// RB + up   → lift right arm (right_target = 1.57 rad)
// RB + down → lower right arm(right_target = 0.0  rad)
REGISTER_OBSERVATION(arm_commands)
{
    auto & joystick = env->robot->data.joystick;
    static auto cfg = env->cfg["commands"]["arm_pose"]["ranges"];

    static float left_target  = cfg["left_shoulder_pitch"][0].as<float>();
    static float right_target = cfg["right_shoulder_pitch"][0].as<float>();

    if (joystick->LB.pressed && joystick->up.on_pressed)
        left_target = cfg["left_shoulder_pitch"][1].as<float>();
    if (joystick->LB.pressed && joystick->down.on_pressed)
        left_target = cfg["left_shoulder_pitch"][0].as<float>();
    if (joystick->RB.pressed && joystick->up.on_pressed)
        right_target = cfg["right_shoulder_pitch"][1].as<float>();
    if (joystick->RB.pressed && joystick->down.on_pressed)
        right_target = cfg["right_shoulder_pitch"][0].as<float>();

    return std::vector<float>{left_target, right_target};
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}