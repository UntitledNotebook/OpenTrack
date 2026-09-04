#pragma once

#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <glog/logging.h>
#include <mutex>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "unitree/common/thread/thread.hpp"
#include "unitree/idl/hg/LowCmd_.hpp"
#include "unitree/idl/hg/LowState_.hpp"

#include "unitree/common/time/time_tool.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

#include "gamepad.hpp"
#include "officialize_logger.hpp"
#include "robot_interface.hpp"
#include "state_machine.hpp"

// ControlFSM States
#include "fsm_basic_controller.hpp"
#include "fsm_tracker_controller.hpp"
#include "fsm_loco_controller.hpp"
#include "fsm_stand_controller.hpp"

using namespace unitree::common;
using namespace unitree::robot;
namespace fs = std::filesystem;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

// FSM controller registry. controller_mapping[group][slot] maps:
//   group : mode_index (0..3, cycled by gamepad B)
//   slot  : 0..3 = dpad, 4..7 = L1+dpad, 8..11 = R1+dpad,
//           12..15 = L2+dpad, 16..19 = R2+dpad
struct FSMStateList
{
    BasicUserController *invalid;
    FsmStandController *ctrl_stand;
    FsmLocoController *ctrl_locomotion;
    std::vector<std::vector<BasicUserController *>> controller_mapping = std::vector<std::vector<BasicUserController *>>(4, std::vector<BasicUserController *>(20, nullptr));
};


enum class DANCE_ORDER
{
    KDANCE1,

    KDANCE_DONE
};

static inline const char *StateName(STATES s)
{
    switch (s)
    {
        case STATES::DAMPING: return "DAMPING";
        case STATES::STAND:   return "STAND";
        case STATES::DANCE:   return "DANCE";
        case STATES::LOCO:    return "LOCO";
    }
    return "UNKNOWN";
}

class RobotController
{
public:
    // 停止标志
    std::atomic<bool> stop_requested_;

    RobotController() {}

    RobotController(fs::path &log_file_name)
    {
        // set log file
        log_file = std::ofstream(log_file_name, std::ios::binary);

        _stateList.invalid = nullptr;
        _stateList.ctrl_stand = new FsmStandController();
        _stateList.ctrl_locomotion = new FsmLocoController();

        struct MotionBinding
        {
            const char *policy;
            const char *motion;
        };

        // Group 0: locally installed tracker and converted reference motions.
        // Up, Down, Left, Right, L1+Up. Other slots remain empty.
        const std::array<MotionBinding, 5> motion_bindings = {{
            {"G1TrackingGeneralDR4010", "rr_stand_still"},
            {"G1TrackingGeneralDR4010", "rr_walk_slow"},
            {"G1TrackingGeneralDR4010", "rr_two_foot_jump"},
            {"G1TrackingGeneralDR4010", "rr_hurdle_jump"},
        }};

        for (size_t i = 0; i < motion_bindings.size(); ++i)
        {
            const int group = static_cast<int>(i / 20);
            const int slot = static_cast<int>(i % 20);
            _stateList.controller_mapping[group][slot] = new FsmTrackerController(
                motion_bindings[i].policy, motion_bindings[i].motion);
        }

        curr_fsm_ctrl_ptr = _stateList.invalid;
        dance_order = DANCE_ORDER::KDANCE1;

        stop_requested_ = false;

#if ENABLE_DANCE_TORQUE_PROJECTION
        dance_torque_limit_ = DefaultDanceTorqueLimit();
        bool torque_limit_loaded = LoadDanceTorqueLimitFromYaml();
        if (!torque_limit_loaded)
        {
            LOG(WARNING) << "Failed to load TORQUE_LIMIT from ../../storage/g1_tracking_constant.yaml, "
                         << "fallback to built-in defaults";
        }

        dance_torque_projection_enabled_ = false;
        if (const char *env = std::getenv("G1_ENABLE_DANCE_TORQUE_PROJECTION"))
        {
            std::string value(env);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (value == "1" || value == "true" || value == "on" || value == "yes")
            {
                dance_torque_projection_enabled_ = true;
            }
        }
#endif

        LogDanceProjectionConfig();

        // Push the loaded policy/motion table into selfcheck.log [policies] section.
        {
            auto &lg = OfficialLogger::Instance();
            lg.SelfcheckSection("policies");
            for (int g = 0; g < static_cast<int>(_stateList.controller_mapping.size()); ++g)
            {
                for (int s = 0; s < static_cast<int>(_stateList.controller_mapping[g].size()); ++s)
                {
                    auto *ctrl = _stateList.controller_mapping[g][s];
                    auto *tracker = dynamic_cast<FsmTrackerController *>(ctrl);
                    if (tracker == nullptr) continue;
                    std::string key = "m" + std::to_string(g) + "_s" + std::to_string(s);
                    std::string value = std::string("policy=") + tracker->GetPolicyName() +
                                        ", motion=" + tracker->GetMotionName() +
                                        ", checkpoint_dir=" + tracker->GetPolicyCheckpointDir();
                    lg.Selfcheck(key, value);
                }
            }
        }
    }
    ~RobotController()
    {
        if (stop_requested_)
            control_thread_ptr.reset();
        if (stop_requested_)
            low_cmd_write_thread_ptr.reset();
    }

    DANCE_ORDER dance_order;

    void LoadParam(fs::path &param_folder)
    {
        if (curr_fsm_ctrl_ptr == nullptr)
        {
            curr_fsm_ctrl_ptr = new FsmStandController();
            curr_fsm_ctrl_ptr->Reset();
        }
        curr_fsm_ctrl_ptr->LoadParam(param_folder);
    }
    std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;
    bool motion_switcher_disabled_ = false;
    bool audio_client_disabled_ = false;
    void InitDdsModel(const std::string &networkInterface = "")
    {
        // Initialize the DDS factory (singleton).
        ChannelFactory::Instance()->Init(0, networkInterface);

        auto &lg = OfficialLogger::Instance();
        lg.SelfcheckSection("hardware_or_sim");

        // Env-gated disable for simulation (no AudioClient / MotionSwitcher servers):
        //   DEPLOY_DISABLE_MOTION_SWITCHER=1  -> skip msc_ init + CheckMode polling
        //   DEPLOY_DISABLE_AUDIO_CLIENT=1     -> skip AudioClient init
        if (const char *env = std::getenv("DEPLOY_DISABLE_MOTION_SWITCHER"))
        {
            std::string v(env);
            std::transform(v.begin(), v.end(), v.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            motion_switcher_disabled_ = (v == "1" || v == "true" || v == "on" || v == "yes");
        }
        if (const char *env = std::getenv("DEPLOY_DISABLE_AUDIO_CLIENT"))
        {
            std::string v(env);
            std::transform(v.begin(), v.end(), v.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            audio_client_disabled_ = (v == "1" || v == "true" || v == "on" || v == "yes");
        }

        lg.Selfcheck("audio_client", audio_client_disabled_ ? "disabled" : "enabled");
        lg.Selfcheck("motion_switcher", motion_switcher_disabled_ ? "disabled" : "enabled");

        if (!audio_client_disabled_)
        {
            client = new unitree::robot::g1::AudioClient();
            client->Init();
            client->SetTimeout(2.0f);

            int32_t ret = client->SetVolume(80);
            lg.Selfcheck("audio_setvolume_ret", std::to_string(ret));
        }

        if (!motion_switcher_disabled_)
        {
            msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
            msc_->SetTimeout(5.0f);
            msc_->Init();
            std::string form, name;
            while (msc_->CheckMode(form, name), !name.empty())
            {
                if (msc_->ReleaseMode())
                    LOG(ERROR) << "Failed to switch to Release Mode";
                sleep(5);
            }
        }

        // Low-level command publisher (motor targets).
        lowcmd_publisher.reset(new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher->InitChannel();

        // Low-level state subscriber (robot feedback).
        lowstate_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber->InitChannel(
            std::bind(&RobotController::LowStateMessageHandler, this, std::placeholders::_1), 1);
    }

    void StartControl()
    {
        // Wait for R2 before spinning up the control thread.
        std::chrono::milliseconds duration(20);
        LOG(INFO) << "Press R2 to start!";
        const auto wait_start = std::chrono::steady_clock::now();
        auto last_wait_log = wait_start;
        while (true)
        {
            InteprateGamePad();

            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_wait_log).count() >= 5)
            {
                const auto &remote = state.wireless_remote();
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - wait_start).count();
                LOG(INFO) << "[WAIT_R2] elapsed=" << elapsed << "s remote23=("
                          << static_cast<int>(remote[2]) << "," << static_cast<int>(remote[3]) << ")";
                last_wait_log = now;
            }

            if (gamepad.R2.on_press || gamepad.R2.pressed)
            {
                break;
            }

            std::this_thread::sleep_for(duration);
        }

        // First lowstate age in the selfcheck file.
        {
            auto &lg = OfficialLogger::Instance();
            lg.SelfcheckSection("dds");
            lg.Selfcheck("first_lowstate_rx_count", std::to_string(lowstate_rx_count_));
        }

        LOG(INFO) << "--------------- Start ---------------";

        // 50Hz control thread (20000us period).
        control_thread_ptr =
            CreateRecurrentThreadEx("ctrl", UT_CPU_ID_NONE, 20000, &RobotController::ControlStep, this);

        while (true)
        {
            if (stop_requested_)
            {
                LOG(INFO) << "[EXIT] reason=signal code=0";
                LOG(INFO) << "Exit signal received. Exiting program.";
                OfficialLogger::Instance().Event("EXIT", "reason=signal, code=0");
                OfficialLogger::Instance().DanceClose("program_exit");
                break;
            }
        }
    }

protected:
    ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber;
    ThreadPtr low_cmd_write_thread_ptr, control_thread_ptr;
    unitree_hg::msg::dds_::LowState_ state;
    int32_t mode_index = 0;

    Gamepad gamepad;
    REMOTE_DATA_RX rx;

    // Control state machine.
    SimpleStateMachine state_machine;
    FSMStateList _stateList;
    BasicUserController *curr_fsm_ctrl_ptr = nullptr;
    unitree::robot::g1::AudioClient* client = nullptr;

    RobotInterface robot_interface;

    // ------------------------------------------------------------------
    // Smooth FSM-switch interpolation (motor protection on real hardware)
    //
    // When swapping the active controller for dance<->dance and dance<->X
    // (Loco) transitions, we linearly blend jpos_des/kp/kd from the
    // last-applied command into the new controller's first outputs over
    // `transition_total_steps_` 50Hz cycles. This avoids the instantaneous
    // setpoint jump that previously kicked the motors at switch time.
    // ------------------------------------------------------------------
    std::array<float, 29> last_jpos_des_{};
    std::array<float, 29> last_kp_{};
    std::array<float, 29> last_kd_{};
    bool have_last_command_ = false;
    bool transition_active_ = false;
    int transition_step_ = 0;
    int transition_total_steps_ = 0;
    static constexpr float kSwitchBlendSeconds = 1.0f;

    std::ofstream log_file;
#if ENABLE_DANCE_TORQUE_PROJECTION
    std::array<float, 29> dance_torque_limit_ = {};
    uint64_t dance_projection_cycle_count_ = 0;
    bool dance_torque_projection_enabled_ = false;
#endif

    uint64_t ctrl_dt_micro_sec = 2000;

    // Runtime diagnostics counters (low-frequency logging to avoid flooding).
    uint64_t lowstate_rx_count_ = 0;
    uint64_t lowcmd_get_count_ = 0;
    uint64_t lowcmd_tx_count_ = 0;
    uint64_t control_cycle_count_ = 0;

    // Cached previous control state for dance CSV close-on-exit detection.
    STATES prev_state_ = STATES::DAMPING;

    // Set once from $G1_LOG_VERBOSE; gates [DDS_RX]/[DDS_TX] heartbeat lines.
    bool verbose_logging_ = [] {
        const char *env = std::getenv("G1_LOG_VERBOSE");
        if (!env) return false;
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return v == "1" || v == "true" || v == "on" || v == "yes";
    }();

    std::vector<float> compute_time;

private:
    static constexpr uint64_t kDebugRxLogInterval = 500;
    static constexpr uint64_t kDebugTxLogInterval = 200;
    static constexpr uint64_t kMotionModeCheckInterval = 100;

    void LogDanceProjectionConfig() const
    {
        auto &lg = OfficialLogger::Instance();
#if ENABLE_DANCE_TORQUE_PROJECTION
        const char *compile = "ON";
        const char *runtime = dance_torque_projection_enabled_ ? "ON" : "OFF";
#else
        const char *compile = "OFF";
        const char *runtime = "OFF";
#endif
        // Preserve the legacy [PROJECTION_CFG] line for start_*.sh grep compatibility.
        LOG(INFO) << "[PROJECTION_CFG] compile=" << compile << ", runtime=" << runtime
                  << ", env=G1_ENABLE_DANCE_TORQUE_PROJECTION(1/true/on/yes)";
        lg.SelfcheckSection("build");
        lg.Selfcheck("projection_compile_runtime_check",
                     std::string(compile) + "/" + runtime);
    }

    void LowStateMessageHandler(const void *message)
    {
        state = *(unitree_hg::msg::dds_::LowState_ *)message;
        robot_interface.LoadState(state);

        ++lowstate_rx_count_;
        if (!verbose_logging_) return;
        if ((lowstate_rx_count_ % kDebugRxLogInterval) != 0) return;

        const auto &motors = state.motor_state();
        const auto &remote = state.wireless_remote();
        if (!motors.empty())
        {
            const auto &m0 = motors[0];
            LOG(INFO) << "[DDS_RX] lowstate cnt=" << lowstate_rx_count_
                      << ", mode_machine=" << static_cast<int>(state.mode_machine())
                      << ", m0(q,dq,tau_est)=(" << m0.q() << "," << m0.dq() << "," << m0.tau_est() << ")"
                      << ", remote23=(" << static_cast<int>(remote[2]) << ","
                      << static_cast<int>(remote[3]) << ")";
        }
    }

    void InteprateGamePad()
    {
        // update gamepad
        memcpy(rx.buff, &state.wireless_remote()[0], 40);
        gamepad.update(rx.RF_RX);
    }

    void LowCmdwriteHandler()
    {
        unitree_hg::msg::dds_::LowCmd_ cmd;
        bool bret = robot_interface.GetLowCmd(cmd);

        ++lowcmd_get_count_;
        if (!bret)
        {
            if (verbose_logging_ && (lowcmd_get_count_ % kDebugTxLogInterval) == 0)
            {
                LOG(WARNING) << "[GetLowCmd] failed cnt=" << lowcmd_get_count_
                             << ", state=" << static_cast<int>(state_machine.state);
            }
            return;
        }

        lowcmd_publisher->Write(cmd);
        ++lowcmd_tx_count_;

        if (verbose_logging_ && (lowcmd_tx_count_ % kDebugTxLogInterval) == 0)
        {
            const auto &motors = cmd.motor_cmd();
            if (!motors.empty())
            {
                const auto &m0 = motors[0];
                LOG(INFO) << "[DDS_TX] lowcmd cnt=" << lowcmd_tx_count_
                          << ", state=" << static_cast<int>(state_machine.state)
                          << ", m0(q,kp,kd,tau)=(" << m0.q() << "," << m0.kp() << ","
                          << m0.kd() << "," << m0.tau() << ")";
            }
        }
    }

    // 50Hz cmd writer thread (currently unused; LowCmdwriteHandler is invoked
    // synchronously from ControlStep instead).
    void StartSendCmd()
    {
        low_cmd_write_thread_ptr =
            CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 20000, &RobotController::LowCmdwriteHandler, this);
    }

    void LogRuntimeEvent(const std::string &event_type, const std::string &event_text)
    {
        OfficialLogger::Instance().Event(event_type, event_text);
    }

    // Begin a linear blend from the last-applied (jpos_des, kp, kd) to the
    // next controller's outputs. No-op if we never produced a command yet.
    void BeginSwitchTransition(const std::string &reason,
                               float seconds = kSwitchBlendSeconds)
    {
        if (!have_last_command_)
        {
            return;
        }
        // ControlStep runs at 50Hz (20000us period) regardless of
        // ctrl_dt_micro_sec which tracks the lowcmd write interval.
        constexpr float kControlPeriodSec = 0.02f;
        int steps = static_cast<int>(std::lround(seconds / kControlPeriodSec));
        if (steps < 1)
        {
            steps = 1;
        }
        transition_total_steps_ = steps;
        transition_step_ = 0;
        transition_active_ = true;
        LogRuntimeEvent(
            "FSM_BLEND_BEGIN",
            "reason=" + reason + ", steps=" + std::to_string(steps) +
                ", seconds=" + std::to_string(seconds));
    }

    // Blend mc.{jpos_des,kp,kd} from last_* into mc.* over the active
    // transition window. Must be called AFTER the new controller has
    // populated mc and BEFORE motor_command_buffer_.SetData().
    void ApplySwitchTransition(MotorCommand &mc)
    {
        if (!transition_active_)
        {
            return;
        }
        ++transition_step_;
        const float alpha =
            std::min(1.0f, static_cast<float>(transition_step_) /
                               static_cast<float>(transition_total_steps_));
        const float one_minus_alpha = 1.0f - alpha;
        for (size_t i = 0; i < mc.jpos_des.size(); ++i)
        {
            mc.jpos_des[i] = one_minus_alpha * last_jpos_des_[i] + alpha * mc.jpos_des[i];
            mc.kp[i]       = one_minus_alpha * last_kp_[i]       + alpha * mc.kp[i];
            mc.kd[i]       = one_minus_alpha * last_kd_[i]       + alpha * mc.kd[i];
        }
        if (transition_step_ >= transition_total_steps_)
        {
            transition_active_ = false;
            LogRuntimeEvent("FSM_BLEND_END", "steps=" + std::to_string(transition_step_));
        }
    }

#if ENABLE_DANCE_TORQUE_PROJECTION
    std::array<float, 29> DefaultDanceTorqueLimit() const
    {
        return {
            88.0f, 139.0f, 88.0f, 139.0f, 50.0f, 50.0f,
            88.0f, 139.0f, 88.0f, 139.0f, 50.0f, 50.0f,
            88.0f, 50.0f, 50.0f,
            25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 5.0f, 5.0f,
            25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 5.0f, 5.0f};
    }

    bool LoadDanceTorqueLimitFromYaml()
    {
        auto &lg = OfficialLogger::Instance();
        lg.SelfcheckSection("torque_limit");

        const fs::path yaml_path = fs::current_path() / "../../storage/g1_tracking_constant.yaml";
        lg.Selfcheck("path", yaml_path.string());

        if (!fs::exists(yaml_path))
        {
            lg.Selfcheck("source", "defaults");
            lg.Selfcheck("reason", "yaml_not_found");
            return false;
        }

        YAML::Node config;
        try
        {
            config = YAML::LoadFile(yaml_path.string());
        }
        catch (const std::exception &e)
        {
            lg.Selfcheck("source", "defaults");
            lg.Selfcheck("reason", std::string("yaml_parse_error: ") + e.what());
            return false;
        }

        const YAML::Node torque_limit_node = config["TORQUE_LIMIT"];
        if (!torque_limit_node || !torque_limit_node.IsSequence())
        {
            lg.Selfcheck("source", "defaults");
            lg.Selfcheck("reason", "TORQUE_LIMIT_missing_or_invalid");
            return false;
        }

        if (torque_limit_node.size() != dance_torque_limit_.size())
        {
            lg.Selfcheck("source", "defaults");
            lg.Selfcheck("reason", "size_mismatch_expected_" + std::to_string(dance_torque_limit_.size()) +
                                       "_got_" + std::to_string(torque_limit_node.size()));
            return false;
        }

        for (size_t i = 0; i < dance_torque_limit_.size(); ++i)
        {
            dance_torque_limit_[i] = torque_limit_node[i].as<float>();
        }
        lg.Selfcheck("source", "yaml");
        lg.Selfcheck("count", std::to_string(dance_torque_limit_.size()));
        return true;
    }

    struct DanceTorqueProjectionStats
    {
        int clipped_joint_count = 0;
        float max_abs_torque_delta = 0.0f;
    };

    bool ShouldApplyDanceTorqueProjection() const
    {
        if (!dance_torque_projection_enabled_)
        {
            return false;
        }

        if (state_machine.state != STATES::DANCE)
        {
            return false;
        }

        return dynamic_cast<FsmTrackerController *>(curr_fsm_ctrl_ptr) != nullptr;
    }

    DanceTorqueProjectionStats ApplyDanceTorqueProjection(
        const std::array<float, 29> &q_actual,
        const std::array<float, 29> &dq_actual,
        const std::array<float, 29> &kp,
        const std::array<float, 29> &kd,
        std::array<float, 29> &q_des) const
    {
        static constexpr float kKpEps = 1e-5f;
        static constexpr float kClipEps = 1e-6f;

        DanceTorqueProjectionStats stats;
        for (size_t i = 0; i < q_des.size(); ++i)
        {
            const float kp_i = kp[i];
            const float kd_i = kd[i];
            const float q_i = q_actual[i];
            const float dq_i = dq_actual[i];
            const float q_des_i = q_des[i];
            const float tau_limit = dance_torque_limit_[i];

            if (!std::isfinite(kp_i) || !std::isfinite(kd_i) || !std::isfinite(q_i) || !std::isfinite(dq_i) ||
                !std::isfinite(q_des_i) || !std::isfinite(tau_limit) || tau_limit <= 0.0f)
            {
                continue;
            }

            if (std::fabs(kp_i) < kKpEps)
            {
                continue;
            }

            const float tau_req = kp_i * (q_des_i - q_i) + kd_i * (0.0f - dq_i);
            const float tau_clip = std::clamp(tau_req, -tau_limit, tau_limit);
            const float abs_tau_delta = std::fabs(tau_req - tau_clip);

            if (abs_tau_delta > kClipEps)
            {
                ++stats.clipped_joint_count;
                stats.max_abs_torque_delta = std::max(stats.max_abs_torque_delta, abs_tau_delta);
            }

            q_des[i] = q_i + (tau_clip + kd_i * dq_i) / kp_i;
        }

        return stats;
    }
#endif

    void EmitFsmEvent(const std::string &btn, STATES from, STATES to, bool ok,
                      const std::string &reason = std::string(),
                      int slot = -1,
                      const std::string &policy = std::string(),
                      const std::string &ckpt_dir = std::string(),
                      const std::string &ckpt_onnx = std::string(),
                      const std::string &motion = std::string())
    {
        std::ostringstream oss;
        oss << "btn=" << btn
            << " from=" << StateName(from)
            << " to=" << StateName(to)
            << " mode_index=" << mode_index
            << " slot=" << slot
            << " policy=" << (policy.empty() ? "-" : policy)
            << " checkpoint_dir=" << (ckpt_dir.empty() ? "-" : ckpt_dir)
            << " checkpoint_onnx=" << (ckpt_onnx.empty() ? "-" : ckpt_onnx)
            << " motion=" << (motion.empty() ? "-" : motion)
#if ENABLE_DANCE_TORQUE_PROJECTION
            << " projection_compile=ON"
            << " projection_runtime=" << (dance_torque_projection_enabled_ ? "ON" : "OFF")
#else
            << " projection_compile=OFF projection_runtime=OFF"
#endif
            << " result=" << (ok ? "OK" : "REJECTED");
        if (!ok) oss << " reason=" << (reason.empty() ? "blocked" : reason);
        LOG(INFO) << "[FSM] " << oss.str();
        OfficialLogger::Instance().Event("FSM", oss.str());
    }

    bool TrySwitchTrackerSlot(int32_t slot, const std::string &btn = "dance")
    {
        const STATES from = state_machine.state;

        if (mode_index < 0 || mode_index >= static_cast<int32_t>(_stateList.controller_mapping.size()))
        {
            EmitFsmEvent(btn, from, STATES::DANCE, false, "mode_index_oob", slot);
            return false;
        }

        if (slot < 0 || slot >= static_cast<int32_t>(_stateList.controller_mapping[mode_index].size()))
        {
            EmitFsmEvent(btn, from, STATES::DANCE, false, "slot_oob", slot);
            return false;
        }

        BasicUserController *target = _stateList.controller_mapping[mode_index][slot];
        if (target == nullptr)
        {
            EmitFsmEvent(btn, from, STATES::DANCE, false, "slot_empty", slot);
            return false;
        }

        if (!state_machine.toDance())
        {
            EmitFsmEvent(btn, from, STATES::DANCE, false, "transition_blocked", slot);
            return false;
        }

        BeginSwitchTransition("to_dance_slot_" + std::to_string(slot));

        curr_fsm_ctrl_ptr = target;
        curr_fsm_ctrl_ptr->Reset();

        FsmTrackerController *tracker = dynamic_cast<FsmTrackerController *>(target);
        std::string policy, ckpt_dir, ckpt_onnx, motion;
        if (tracker != nullptr)
        {
            policy    = tracker->GetPolicyName();
            ckpt_dir  = tracker->GetPolicyCheckpointDir();
            ckpt_onnx = tracker->GetPolicyOnnxPath();
            motion    = tracker->GetMotionName();
        }

        EmitFsmEvent(btn, from, STATES::DANCE, true, std::string(), slot,
                     policy, ckpt_dir, ckpt_onnx, motion);

        LogRuntimeEvent(
            "DANCE_START",
            "mode_index=" + std::to_string(mode_index) +
                ", slot=" + std::to_string(slot) +
                ", policy=" + (policy.empty() ? "<unknown>" : policy) +
                ", checkpoint_dir=" + (ckpt_dir.empty() ? "<unknown>" : ckpt_dir) +
                ", checkpoint_onnx=" + (ckpt_onnx.empty() ? "<unknown>" : ckpt_onnx) +
                ", motion=" + (motion.empty() ? "<unknown>" : motion));

        // Open per-dance CSV/meta for the new dance.
        OfficialLogger::DanceMeta meta;
        meta.mode_index = mode_index;
        meta.slot       = slot;
        meta.policy_name      = policy;
        meta.checkpoint_dir   = ckpt_dir;
        meta.checkpoint_onnx  = ckpt_onnx;
        meta.motion_name      = motion;
#if ENABLE_DANCE_TORQUE_PROJECTION
        meta.projection_compile = "ON";
        meta.projection_runtime = dance_torque_projection_enabled_ ? "ON" : "OFF";
#else
        meta.projection_compile = "OFF";
        meta.projection_runtime = "OFF";
#endif
        // Close any previously-open dance CSV (e.g. dance->dance switch).
        if (OfficialLogger::Instance().DanceIsOpen())
        {
            OfficialLogger::Instance().DanceClose("switch_to_new_dance");
        }
        OfficialLogger::Instance().DanceOpen(meta);

        return true;
    }

// UpdateStateMachine parses one gamepad event per call (if/else if chain;
    // earlier branches take priority). Buttons:
    //   F2          -> state_machine.Stop()
    //   B           -> cycle mode_index, English TTS announcement
    //   A           -> toStand()
    //   X           -> toLoco()
    //   dpad / L1+ / R1+ / L2+ / R2+ -> dance slot 0..3 / 4..7 / 8..11 / 12..15 / 16..19
    // Each successful transition emits a [FSM] event line; each rejected
    // transition also emits a [FSM] line with result=REJECTED and a reason.
    void UpdateStateMachine()
    {
        const STATES from = state_machine.state;

        if (gamepad.F2.pressed)
        {
            state_machine.Stop();
            EmitFsmEvent("F2", from, state_machine.state, true, "stop");
        }
        else if (gamepad.B.on_press)
        {
            // Cycle only groups with registered trackers. A single-group setup
            // stays in group 0 instead of selecting unavailable policy groups.
            const int group_count = static_cast<int>(_stateList.controller_mapping.size());
            for (int offset = 1; offset <= group_count; ++offset)
            {
                const int candidate = (mode_index + offset) % group_count;
                const auto &slots = _stateList.controller_mapping[candidate];
                if (std::any_of(slots.begin(), slots.end(),
                                [](const BasicUserController *ctrl) { return ctrl != nullptr; }))
                {
                    mode_index = candidate;
                    break;
                }
            }
            const std::string voice_text = "Tracker group " + std::to_string(mode_index + 1) + ".";

            int32_t tts_ret = -1;
            if (!voice_text.empty() && client != nullptr)
            {
                tts_ret = client->TtsMaker(voice_text, 1);  // speaker_id=1 -> English TTS
            }

            LOG(INFO) << "[MODE] mode_index=" << mode_index
                      << " voice=\"" << voice_text << "\""
                      << " tts_ret=" << tts_ret;
            OfficialLogger::Instance().Event(
                "MODE",
                "mode_index=" + std::to_string(mode_index) +
                    ", voice=\"" + voice_text + "\", tts_ret=" + std::to_string(tts_ret));
        }
        else if (gamepad.A.on_press)
        {
            if (state_machine.toStand())
            {
                curr_fsm_ctrl_ptr = _stateList.ctrl_stand;
                curr_fsm_ctrl_ptr->Reset();
                EmitFsmEvent("A", from, STATES::STAND, true);
            }
            else
            {
                EmitFsmEvent("A", from, STATES::STAND, false, "transition_blocked");
            }
        }
        else if (gamepad.X.on_press)
        {
            if (state_machine.toLoco())
            {
                BeginSwitchTransition("X_to_loco");
                curr_fsm_ctrl_ptr = _stateList.ctrl_locomotion;
                curr_fsm_ctrl_ptr->Reset();
                EmitFsmEvent("X", from, STATES::LOCO, true);
            }
            else
            {
                EmitFsmEvent("X", from, STATES::LOCO, false, "transition_blocked");
            }
        }
        // Dance slots 4..7  via L1 + dpad
        else if (gamepad.L1.pressed) {
            if      (gamepad.up.on_press)    TrySwitchTrackerSlot(4,  "L1+up");
            else if (gamepad.down.on_press)  TrySwitchTrackerSlot(5,  "L1+down");
            else if (gamepad.left.on_press)  TrySwitchTrackerSlot(6,  "L1+left");
            else if (gamepad.right.on_press) TrySwitchTrackerSlot(7,  "L1+right");
        }
        // Dance slots 8..11 via R1 + dpad
        else if (gamepad.R1.pressed) {
            if      (gamepad.up.on_press)    TrySwitchTrackerSlot(8,  "R1+up");
            else if (gamepad.down.on_press)  TrySwitchTrackerSlot(9,  "R1+down");
            else if (gamepad.left.on_press)  TrySwitchTrackerSlot(10, "R1+left");
            else if (gamepad.right.on_press) TrySwitchTrackerSlot(11, "R1+right");
        }
        // Dance slots 12..15 via L2 + dpad
        else if (gamepad.L2.pressed) {
            if      (gamepad.up.on_press)    TrySwitchTrackerSlot(12, "L2+up");
            else if (gamepad.down.on_press)  TrySwitchTrackerSlot(13, "L2+down");
            else if (gamepad.left.on_press)  TrySwitchTrackerSlot(14, "L2+left");
            else if (gamepad.right.on_press) TrySwitchTrackerSlot(15, "L2+right");
        }
        // Dance slots 16..19 via R2 + dpad
        else if (gamepad.R2.pressed) {
            if      (gamepad.up.on_press)    TrySwitchTrackerSlot(16, "R2+up");
            else if (gamepad.down.on_press)  TrySwitchTrackerSlot(17, "R2+down");
            else if (gamepad.left.on_press)  TrySwitchTrackerSlot(18, "R2+left");
            else if (gamepad.right.on_press) TrySwitchTrackerSlot(19, "R2+right");
        }
        // Dance slots 0..3 via plain dpad
        else {
            if      (gamepad.up.on_press)    TrySwitchTrackerSlot(0, "up");
            else if (gamepad.down.on_press)  TrySwitchTrackerSlot(1, "down");
            else if (gamepad.left.on_press)  TrySwitchTrackerSlot(2, "left");
            else if (gamepad.right.on_press) TrySwitchTrackerSlot(3, "right");
        }
    }

    // ControlStep runs at 50 Hz on the recurrent ctrl thread.
    void ControlStep()
    {
        ++control_cycle_count_;
        if ((control_cycle_count_ % kMotionModeCheckInterval) == 0 && msc_ && !motion_switcher_disabled_)
        {
            std::string form, name;
            msc_->CheckMode(form, name);
            if (!name.empty())
            {
                LOG(WARNING) << "[MOTION_SWITCHER] mode occupied form=" << form << ", name=" << name;
            }
        }

        // F1 -> request exit.
        if (gamepad.F1.on_press)
        {
            LOG(INFO) << "[EXIT] reason=F1 code=0";
            OfficialLogger::Instance().Event("EXIT", "reason=F1, code=0");
            stop_requested_ = true;
            return;
        }

        if (stop_requested_)
        {
            return;
        }

        // update state
        InteprateGamePad();
        UpdateStateMachine(); // 更新获取机器人状态机改变枚举值 并调用对应状态初始的回调函数初始化kpkd init_pos等

        // select control modes according to the state machine
        auto start = std::chrono::high_resolution_clock::now();
        if (state_machine.state == STATES::DAMPING)
        {
            const std::shared_ptr<const MotorState> ms = robot_interface.motor_state_buffer_.GetData();
            MotorCommand mc;
            mc.jpos_des = {-0.1, 0,   0, 0.3,  -0.2, 0, -0.1, 0,   0,    0.3, -0.2, 0, 0, 0, 0,
                           0.2,  0.3, 0, 1.28, 0,    0, 0,    0.2, -0.3, 0,   1.28, 0, 0, 0};
            mc.jvel_des.fill(0.0);
            mc.kd.fill(10);
            mc.kp.fill(0);
            mc.tau_ff.fill(0.0);
            mc.mode_machine = ms->mode_machine;
            // Snapshot so any subsequent FSM activation can blend from DAMPING.
            last_jpos_des_ = mc.jpos_des;
            last_kp_ = mc.kp;
            last_kd_ = mc.kd;
            have_last_command_ = true;
            robot_interface.motor_command_buffer_.SetData(mc);
        }
        else if (state_machine.state > STATES::DAMPING)  // STAND, DANCE, LOCO
        {
            // Shared logic for stand / dance / loco: curr_fsm_ctrl_ptr is swapped.
            curr_fsm_ctrl_ptr->GetInput(robot_interface, gamepad);
            curr_fsm_ctrl_ptr->Calculate();
            const std::shared_ptr<const MotorState> ms = robot_interface.motor_state_buffer_.GetData();
            MotorCommand mc;
            mc.jpos_des = curr_fsm_ctrl_ptr->jpos_des;
            mc.jvel_des.fill(0.0);
            mc.kd = curr_fsm_ctrl_ptr->kd;
            mc.kp = curr_fsm_ctrl_ptr->kp;
            mc.tau_ff.fill(0.0);
            mc.mode_machine = ms->mode_machine;

            // Snapshot raw policy output BEFORE torque projection + blending.
            const std::array<float, 29> action_raw = mc.jpos_des;

            int tau_clip_count = 0;
#if ENABLE_DANCE_TORQUE_PROJECTION
            if (ShouldApplyDanceTorqueProjection())
            {
                const DanceTorqueProjectionStats stats =
                    ApplyDanceTorqueProjection(ms->jpos, ms->jvel, mc.kp, mc.kd, mc.jpos_des);
                ++dance_projection_cycle_count_;
                tau_clip_count = stats.clipped_joint_count;

                if (stats.clipped_joint_count > 0 && (dance_projection_cycle_count_ % 25 == 0))
                {
                    LogRuntimeEvent(
                        "DANCE_TAU_CLIP",
                        "clipped_joints=" + std::to_string(stats.clipped_joint_count) +
                            ", max_abs_tau_delta=" + std::to_string(stats.max_abs_torque_delta));
                }
            }
#endif

            // Smooth blend across FSM switches.
            ApplySwitchTransition(mc);

            last_jpos_des_ = mc.jpos_des;
            last_kp_ = mc.kp;
            last_kd_ = mc.kd;
            have_last_command_ = true;

            robot_interface.motor_command_buffer_.SetData(mc);

            // Emit one dance CSV row per cycle while DANCE is active.
            if (state_machine.state == STATES::DANCE && OfficialLogger::Instance().DanceIsOpen())
            {
                const std::shared_ptr<const ImuState> imu_ptr = robot_interface.imu_state_buffer_.GetData();
                std::array<float, 4> quat = {1.0f, 0.0f, 0.0f, 0.0f};
                std::array<float, 3> omega = {0.0f, 0.0f, 0.0f};
                if (imu_ptr)
                {
                    quat  = imu_ptr->quat;
                    omega = imu_ptr->gyro;
                }
                const auto &remote = state.wireless_remote();
                OfficialLogger::Instance().DanceRow(
                    ms->jpos, ms->jvel, quat, omega,
                    remote[2], remote[3],
                    action_raw, mc.jpos_des, mc.kp, mc.kd,
                    tau_clip_count);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = end - start;
            auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);

            dance_order = DANCE_ORDER::KDANCE1;

            if (curr_fsm_ctrl_ptr->dance_done_flag == true)
            {
                curr_fsm_ctrl_ptr->dance_done_flag = false;
                state_machine.state = STATES::LOCO;
                BeginSwitchTransition("dance_done_to_loco");
                curr_fsm_ctrl_ptr = _stateList.ctrl_locomotion;
                curr_fsm_ctrl_ptr->Reset();
                EmitFsmEvent("auto", STATES::DANCE, STATES::LOCO, true, "dance_done");
            }
        }

        // Close any open dance CSV when we leave DANCE state.
        if (prev_state_ == STATES::DANCE && state_machine.state != STATES::DANCE)
        {
            OfficialLogger::Instance().DanceClose("state_change");
        }
        prev_state_ = state_machine.state;

        LowCmdwriteHandler();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        compute_time.push_back(duration.count() / 1000.);

        WriteLog();
    }

    void WriteLog()
    {
        if (log_file.is_open())
        {
            auto log = curr_fsm_ctrl_ptr->GetLog();
            for (const auto &v : log)
            {
                log_file << v << " ";
            }
            log_file << std::endl;
        }
    }
};
