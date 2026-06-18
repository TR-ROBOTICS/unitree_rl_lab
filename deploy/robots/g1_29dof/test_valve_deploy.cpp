// Offline smoke test for the valve-turn deploy config: builds the env from
// deploy.yaml, runs one policy inference on a zeroed robot state, and checks the
// observation/action dimensions. No sim, no display, no DDS traffic needed.
#include "isaaclab/envs/manager_based_rl_env.h"
#include "unitree_articulation.h"
#include "State_ValveTurn.h"

#include <unitree/robot/channel/channel_factory.hpp>
#include <cstdio>

// FSMState statics (normally defined in main.cpp). Unused by the env path here,
// but State_ValveTurn.cpp.o references them, so the linker needs definitions.
std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

int main()
{
    unitree::robot::ChannelFactory::Instance()->Init(0, "lo");

    auto lowstate = std::make_shared<LowState_t>();
    auto robot = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(lowstate);

    std::string dir = "config/policy/valve_turn/v5";
    auto env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(dir + "/params/deploy.yaml"), robot);
    env->alg = std::make_unique<isaaclab::OrtRunner>(dir + "/exported/policy.onnx");

    env->reset();
    env->step();

    auto obs = env->observation_manager->compute();
    auto act = env->action_manager->processed_actions();

    printf("OBS_DIM=%zu (expect 30)  ACTION_DIM=%zu (expect 14)\n", obs["obs"].size(), act.size());
    printf("p_now_norm(obs[28])=%.4f  p_des_norm(obs[29])=%.4f\n",
           obs["obs"].size() >= 30 ? obs["obs"][28] : -1.0f,
           obs["obs"].size() >= 30 ? obs["obs"][29] : -1.0f);
    for (size_t i = 0; i < act.size(); ++i) printf("  action[%zu] = %.4f\n", i, act[i]);
    return (obs["obs"].size() == 30 && act.size() == 14) ? 0 : 1;
}
