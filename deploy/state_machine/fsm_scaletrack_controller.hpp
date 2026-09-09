#pragma once

#include "fsm_basic_controller.hpp"

#include <array>
#include <eigen3/Eigen/Geometry>
#include <filesystem>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "gamepad.hpp"
#include "robot_interface.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    // ScaleTrack's Transformer deployment interface.  Unlike the legacy
    // tracker, this controller supplies short proprioceptive/action histories
    // and future rigid-body targets; the ONNX wrapper performs FK, mode
    // masking, and action scaling internally.
    class FsmScaleTrackController : public BasicUserController
    {
    public:
        FsmScaleTrackController(std::string policy_name, std::string data_name);

        void LoadParam(fs::path &param_folder) override;
        void Reset() override;
        void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) override;
        void Calculate() override;
        std::vector<float> GetLog() override;

        const std::string &GetPolicyName() const { return policy_name_; }
        const std::string &GetPolicyCheckpointDir() const { return checkpoint_dir_name_; }
        const std::string &GetPolicyOnnxPath() const { return policy_onnx_path_; }
        const std::string &GetMotionName() const { return data_name_; }
        int GetModeIndex() const { return mode_index_; }
        void SetReferenceAdvanceEnabled(bool enabled) { reference_advance_enabled_ = enabled; }

    private:
        static constexpr int kNumMotors = 29;
        static constexpr int kContext = 3;
        static constexpr int kFuture = 6;
        static constexpr int kSelectedBodies = 14;

        void LoadPolicyMetadata(const fs::path &metadata_path);
        void LoadReference();
        void ValidatePolicyInterface();
        void PushStateHistory();
        void PushActionHistory();
        void BuildFutureTargets();

        Eigen::Quaternionf NormalizeQuaternion(const float *values, const char *name) const;
        Eigen::Quaternionf ReferenceQuaternion(int frame, int body) const;
        Eigen::Vector3f ReferencePosition(int frame, int body) const;
        static Eigen::Quaternionf YawOnly(const Eigen::Quaternionf &quaternion);

        Ort::Env env_;
        Ort::SessionOptions session_options_;
        Ort::AllocatorWithDefaultOptions allocator_;
        std::unique_ptr<Ort::Session> policy_session_;

        std::string policy_name_;
        std::string checkpoint_dir_name_;
        std::string policy_onnx_path_;
        std::string data_name_;
        int mode_index_ = 7;

        std::array<float, kNumMotors> default_dof_pos_{};
        std::array<int64_t, kFuture> future_offsets_{{0, 1, 2, 3, 4, 5}};

        std::vector<float> reference_body_positions_;
        std::vector<float> reference_body_quaternions_;
        int reference_frames_ = 0;
        int inference_counter_ = 0;
        bool reference_advance_enabled_ = true;

        std::array<float, kNumMotors> joint_pos_{};
        std::array<float, kNumMotors> joint_vel_{};
        std::array<float, 4> root_quat_{{1.0f, 0.0f, 0.0f, 0.0f}};
        std::array<float, 3> base_ang_vel_{};

        std::vector<float> root_quat_history_;
        std::vector<float> base_ang_vel_history_;
        std::vector<float> dof_pos_history_;
        std::vector<float> dof_vel_history_;
        std::vector<float> action_history_;
        std::array<float, kNumMotors> previous_action_{};
        bool state_history_initialized_ = false;

        std::vector<float> future_body_positions_base_;
        std::vector<float> future_body_quaternions_base_;
        std::array<int64_t, 1> mode_index_input_{{7}};
        std::array<int64_t, kFuture> time_offsets_input_{{0, 1, 2, 3, 4, 5}};

        Eigen::Quaternionf world_alignment_ = Eigen::Quaternionf::Identity();
        bool world_alignment_initialized_ = false;
    };
} // namespace unitree::common
