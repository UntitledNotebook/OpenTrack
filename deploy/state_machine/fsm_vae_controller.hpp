#pragma once

#include "fsm_basic_controller.hpp"

#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <random>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "gamepad.hpp"
#include "robot_interface.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    // Deploys the unified VAE ONNX exported by Humanoid_Pipeline.  The model
    // takes (auxiliary_state, state, noise) and exposes posterior and prior
    // action outputs.  Posterior + zero noise is the safe deterministic
    // reference-tracking default.
    class FsmVaeController : public BasicUserController
    {
    public:
        FsmVaeController(std::string policy_name, std::string data_name);

        void LoadParam(fs::path &param_folder) override;
        void Reset() override;
        void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) override;
        void Calculate() override;
        std::vector<float> GetLog() override;

        const std::string &GetPolicyName() const { return policy_name_; }
        const std::string &GetPolicyCheckpointDir() const { return checkpoint_dir_name_; }
        const std::string &GetPolicyOnnxPath() const { return policy_onnx_path_; }
        const std::string &GetMotionName() const { return data_name_; }
        const std::string &GetInferenceMode() const { return inference_mode_; }
        float GetNoiseStd() const { return noise_std_; }
        void SetReferenceAdvanceEnabled(bool enabled) { reference_advance_enabled_ = enabled; }
        void SetLastAppliedMotorTargets(const std::array<float, 29> &targets)
        {
            for (int i = 0; i < kNumMotors; ++i)
            {
                last_motor_targets_[i] = targets[i];
            }
            has_last_motor_targets_ = true;
        }

    private:
        static constexpr int kNumMotors = 29;
        static constexpr int kNoiseDim = 32;

        Eigen::Vector3f RpyToGravity(const std::array<float, 3> &rpy) const;
        std::array<float, kNumMotors> ReadYamlArray(const YAML::Node &node) const;
        std::vector<float> ReadYamlFloatVector(const YAML::Node &node) const;
        std::vector<int> ReadYamlIntVector(const YAML::Node &node) const;
        void LoadReference();
        int ObservationWidth(const std::vector<std::string> &names) const;
        std::vector<float> BuildObservation(
            const std::vector<std::string> &names,
            const Eigen::Ref<const Eigen::VectorXf> &ref_qpos,
            const Eigen::Ref<const Eigen::VectorXf> &ref_qvel) const;

        Ort::Env env_;
        Ort::SessionOptions session_options_;
        Ort::AllocatorWithDefaultOptions allocator_;
        std::unique_ptr<Ort::Session> session_;

        std::string policy_name_;
        std::string checkpoint_dir_name_;
        std::string policy_onnx_path_;
        std::string data_name_;
        std::string inference_mode_ = "posterior";

        std::vector<std::string> obs_names_;
        std::vector<std::string> auxiliary_obs_names_;
        std::vector<int> obs_joint_ids_;
        std::vector<float> default_qpos_;

        float action_scale_ = 1.0f;
        float joint_vel_scale_ = 0.05f;
        float dif_joint_vel_scale_ = 0.05f;
        float noise_std_ = 0.0f;
        bool student_use_residual_action_ = false;
        int state_dim_ = 0;
        int auxiliary_state_dim_ = 0;
        int action_dim_ = kNumMotors;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref_qpos_all_;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref_qvel_all_;
        int end_iter_ = 0;
        int inference_counter_ = 0;
        bool reference_advance_enabled_ = true;

        Eigen::VectorXf joint_pos_ = Eigen::VectorXf::Zero(kNumMotors);
        Eigen::VectorXf joint_vel_ = Eigen::VectorXf::Zero(kNumMotors);
        Eigen::VectorXf last_motor_targets_ = Eigen::VectorXf::Zero(kNumMotors);
        Eigen::Vector3f pelvis_gvec_ = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f gyro_ = Eigen::Vector3f::Zero();
        bool has_last_motor_targets_ = false;

        std::mt19937 random_engine_{0};
        std::normal_distribution<float> normal_distribution_{0.0f, 1.0f};
    };
} // namespace unitree::common
