#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_ValveTurn.h"

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/ros2/Point_.hpp>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";
    std::cout << "W / S — raise / lower p_des by 5 PSI\n";

    // Publisher for live p_des updates — valve_pressure_node.py relays to controller.
    constexpr float P_DES_STEP = 5.0f, P_MIN = 15.0f, P_MAX = 200.0f;
    float p_des = 100.0f;
    auto p_des_pub = std::make_shared<unitree::robot::ChannelPublisher<geometry_msgs::msg::dds_::Point_>>("rt/valve/pressure_des_cmd");
    p_des_pub->InitChannel();

    geometry_msgs::msg::dds_::Point_ p_des_msg;
    p_des_msg.y(0.0); p_des_msg.z(0.0);

    std::string last_key = "";
    while (true)
    {
        usleep(90000);  // slightly above keyboard thread's 80ms read window
        auto k = FSMState::keyboard->key();
        if (k == last_key || k.empty()) { last_key = k; continue; }
        last_key = k;
        if (k == "w")      p_des = std::min(P_MAX, p_des + P_DES_STEP);
        else if (k == "s") p_des = std::max(P_MIN, p_des - P_DES_STEP);
        else continue;
        p_des_msg.x(p_des);
        p_des_pub->Write(p_des_msg);
        std::cout << "\r[p_des] " << p_des << " PSI    " << std::flush;
    }
    
    return 0;
}

