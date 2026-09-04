#include "fsm_vae_controller.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

#include <glog/logging.h>

#include "json.hpp"

namespace unitree::common
{
    namespace
    {
        bool IsNumeric(const std::string &value)
        {
            return !value.empty() &&
                   std::all_of(value.begin(), value.end(), [](unsigned char c) {
                       return std::isdigit(c) != 0;
                   });
        }

        std::string Lower(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return value;
        }

        void ValidateTensorShape(
            Ort::Session &session,
            bool input,
            const std::string &name,
            int expected_width,
            Ort::AllocatorWithDefaultOptions &allocator)
        {
            const size_t count = input ? session.GetInputCount() : session.GetOutputCount();
            for (size_t i = 0; i < count; ++i)
            {
                auto allocated_name = input
                    ? session.GetInputNameAllocated(i, allocator)
                    : session.GetOutputNameAllocated(i, allocator);
                if (name != allocated_name.get())
                {
                    continue;
                }

                const auto type_info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
                const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    throw std::runtime_error("VAE tensor '" + name + "' is not float32");
                }
                const auto shape = tensor_info.GetShape();
                if (shape.size() != 2 || (shape[1] > 0 && shape[1] != expected_width))
                {
                    throw std::runtime_error(
                        "VAE tensor '" + name + "' width mismatch: expected " +
                        std::to_string(expected_width));
                }
                return;
            }
            throw std::runtime_error(
                std::string("VAE model is missing ") + (input ? "input '" : "output '") +
                name + "'");
        }
    } // namespace

    FsmVaeController::FsmVaeController(std::string policy_name, std::string data_name)
        : env_(ORT_LOGGING_LEVEL_WARNING, "VAE_ONNXRuntime"),
          policy_name_(std::move(policy_name)),
          data_name_(std::move(data_name))
    {
        const fs::path checkpoints_dir =
            fs::path("../../storage/policy") / policy_name_ / "checkpoints";
        const fs::path config_path = checkpoints_dir / "config.json";
        if (!fs::exists(config_path))
        {
            throw std::runtime_error("VAE config not found: " + config_path.string());
        }

        nlohmann::json config;
        {
            std::ifstream stream(config_path);
            if (!stream)
            {
                throw std::runtime_error("cannot open VAE config: " + config_path.string());
            }
            stream >> config;
        }

        const auto &env_config = config.at("env_config");
        const auto &policy_args = config.at("policy_config").at("policy_args");
        if (policy_args.value("policy_type", std::string()) != "vae")
        {
            throw std::runtime_error("policy_config.policy_args.policy_type must be 'vae'");
        }

        for (const auto &name : env_config.at("obs_keys"))
        {
            obs_names_.push_back(name.get<std::string>());
        }
        for (const auto &name : env_config.at("auxiliary_obs_keys"))
        {
            auxiliary_obs_names_.push_back(name.get<std::string>());
        }

        state_dim_ = policy_args.at("obs_dim").get<int>();
        auxiliary_state_dim_ = policy_args.at("aux_obs_dim").get<int>();
        action_dim_ = policy_args.at("act_dim").get<int>();
        const int noise_dim = policy_args.at("vae_latent_dim").get<int>();
        action_scale_ = env_config.value("action_scale", 1.0f);
        student_use_residual_action_ =
            env_config.value("student_use_residual_action", true);
        const auto obs_scales = env_config.value(
            "obs_scales_config", nlohmann::json::object());
        joint_vel_scale_ = obs_scales.value("joint_vel", 0.05f);
        dif_joint_vel_scale_ = obs_scales.value("dif_joint_vel", joint_vel_scale_);

        if (state_dim_ != 93 || auxiliary_state_dim_ != 58 ||
            action_dim_ != kNumMotors || noise_dim != kNoiseDim)
        {
            throw std::runtime_error(
                "unsupported VAE dimensions; expected state=93, aux=58, noise=32, action=29");
        }
        if (env_config.value("history_len", 0) != 0)
        {
            throw std::runtime_error("VAE deployment currently requires history_len=0");
        }
        if (policy_args.value("output_residual_action", student_use_residual_action_) !=
            student_use_residual_action_)
        {
            throw std::runtime_error(
                "VAE config disagrees on residual-action semantics between policy and environment");
        }

        if (const char *value = std::getenv("G1_VAE_INFERENCE"))
        {
            if (*value != '\0')
            {
                inference_mode_ = Lower(value);
            }
        }
        if (inference_mode_ != "posterior" && inference_mode_ != "prior")
        {
            throw std::runtime_error(
                "G1_VAE_INFERENCE must be 'posterior' or 'prior'");
        }

        if (const char *value = std::getenv("G1_VAE_NOISE_STD"))
        {
            if (*value != '\0')
            {
                noise_std_ = std::stof(value);
            }
        }
        if (!std::isfinite(noise_std_) || noise_std_ < 0.0f || noise_std_ > 5.0f)
        {
            throw std::runtime_error("G1_VAE_NOISE_STD must be finite and in [0, 5]");
        }
        if (const char *value = std::getenv("G1_VAE_NOISE_SEED"))
        {
            if (*value != '\0')
            {
                random_engine_.seed(static_cast<unsigned int>(std::stoul(value)));
            }
        }

        int64_t latest_number = -1;
        fs::path latest_model;
        std::string latest_dir;
        for (const auto &entry : fs::directory_iterator(checkpoints_dir))
        {
            if (!entry.is_directory())
            {
                continue;
            }
            const std::string dirname = entry.path().filename().string();
            if (!IsNumeric(dirname) || !fs::exists(entry.path() / "model.onnx"))
            {
                continue;
            }
            const int64_t number = std::stoll(dirname);
            if (number > latest_number)
            {
                latest_number = number;
                latest_model = entry.path() / "model.onnx";
                latest_dir = dirname;
            }
        }
        if (latest_number >= 0)
        {
            checkpoint_dir_name_ = latest_dir;
            policy_onnx_path_ = latest_model.string();
        }
        else if (fs::exists(checkpoints_dir / "model.onnx"))
        {
            checkpoint_dir_name_ = "flat";
            policy_onnx_path_ = (checkpoints_dir / "model.onnx").string();
        }
        else
        {
            throw std::runtime_error(
                "VAE model not found under " + checkpoints_dir.string());
        }

        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(
            env_, policy_onnx_path_.c_str(), session_options_);

        ValidateTensorShape(*session_, true, "auxiliary_state", auxiliary_state_dim_, allocator_);
        ValidateTensorShape(*session_, true, "state", state_dim_, allocator_);
        ValidateTensorShape(*session_, true, "noise", kNoiseDim, allocator_);
        ValidateTensorShape(*session_, false, "continuous_actions", action_dim_, allocator_);
        ValidateTensorShape(*session_, false, "continuous_actions_prior", action_dim_, allocator_);

        LOG(INFO) << "[VAE] policy=" << policy_name_
                  << ", checkpoint=" << checkpoint_dir_name_
                  << ", model=" << policy_onnx_path_
                  << ", inference=" << inference_mode_
                  << ", noise_std=" << noise_std_
                  << ", action_semantics="
                  << (student_use_residual_action_ ? "residual" : "absolute");
    }

    void FsmVaeController::LoadParam(fs::path &param_folder)
    {
        (void)param_folder;
    }

    std::array<float, FsmVaeController::kNumMotors>
    FsmVaeController::ReadYamlArray(const YAML::Node &node) const
    {
        if (!node || !node.IsSequence() || node.size() != kNumMotors)
        {
            throw std::runtime_error("expected a 29-element YAML sequence");
        }
        std::array<float, kNumMotors> result{};
        for (int i = 0; i < kNumMotors; ++i)
        {
            result[i] = node[i].as<float>();
        }
        return result;
    }

    std::vector<float> FsmVaeController::ReadYamlFloatVector(const YAML::Node &node) const
    {
        if (!node || !node.IsSequence())
        {
            throw std::runtime_error("expected a YAML float sequence");
        }
        std::vector<float> result;
        result.reserve(node.size());
        for (const auto &value : node)
        {
            result.push_back(value.as<float>());
        }
        return result;
    }

    std::vector<int> FsmVaeController::ReadYamlIntVector(const YAML::Node &node) const
    {
        if (!node || !node.IsSequence())
        {
            throw std::runtime_error("expected a YAML integer sequence");
        }
        std::vector<int> result;
        result.reserve(node.size());
        for (const auto &value : node)
        {
            result.push_back(value.as<int>());
        }
        return result;
    }

    void FsmVaeController::LoadReference()
    {
        const fs::path reference_path =
            fs::path("../../storage/data") / data_name_ / "ref_data.onnx";
        if (!fs::exists(reference_path))
        {
            throw std::runtime_error("reference ONNX not found: " + reference_path.string());
        }

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        Ort::Session reference_session(env_, reference_path.c_str(), options);
        const std::array<const char *, 2> output_names = {"qpos", "qvel"};
        auto outputs = reference_session.Run(
            Ort::RunOptions{nullptr}, nullptr, nullptr, 0,
            output_names.data(), output_names.size());

        auto validate = [](Ort::Value &value, const char *name, int min_width) {
            const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 2 || shape[0] <= 0 || shape[1] < min_width)
            {
                throw std::runtime_error(std::string("invalid reference tensor: ") + name);
            }
            return shape;
        };
        const auto qpos_shape = validate(outputs[0], "qpos", 7 + kNumMotors);
        const auto qvel_shape = validate(outputs[1], "qvel", 6 + kNumMotors);
        if (qpos_shape[0] != qvel_shape[0])
        {
            throw std::runtime_error("reference qpos/qvel frame counts differ");
        }

        ref_qpos_all_ = Eigen::Map<
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            outputs[0].GetTensorMutableData<float>(), qpos_shape[0], qpos_shape[1]);
        ref_qvel_all_ = Eigen::Map<
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            outputs[1].GetTensorMutableData<float>(), qvel_shape[0], qvel_shape[1]);
        end_iter_ = static_cast<int>(qpos_shape[0]);
    }

    int FsmVaeController::ObservationWidth(const std::vector<std::string> &names) const
    {
        int width = 0;
        for (const auto &name : names)
        {
            if (name == "gvec_pelvis" || name == "gyro_pelvis")
            {
                width += 3;
            }
            else if (name == "joint_pos" || name == "joint_vel" ||
                     name == "last_motor_targets" || name == "dif_joint_pos" ||
                     name == "dif_joint_vel")
            {
                width += static_cast<int>(obs_joint_ids_.size());
            }
            else
            {
                throw std::runtime_error("unsupported VAE observation key: " + name);
            }
        }
        return width;
    }

    void FsmVaeController::Reset()
    {
        const YAML::Node config = YAML::LoadFile("../../storage/g1_tracking_constant.yaml");
        kp = ReadYamlArray(config["KPs"][0]);
        kd = ReadYamlArray(config["KDs"][0]);
        default_qpos_ = ReadYamlFloatVector(config["DEFAULT_QPOS"]);
        obs_joint_ids_ = ReadYamlIntVector(config["OBS_JOINT_IDS"]);

        if (default_qpos_.size() != kNumMotors || obs_joint_ids_.size() != kNumMotors)
        {
            throw std::runtime_error("VAE deployment requires 29 default and observed joints");
        }
        std::set<int> unique_ids;
        for (const int id : obs_joint_ids_)
        {
            if (id < 0 || id >= kNumMotors || !unique_ids.insert(id).second)
            {
                throw std::runtime_error("OBS_JOINT_IDS must be a permutation of [0, 28]");
            }
        }
        if (ObservationWidth(obs_names_) != state_dim_ ||
            ObservationWidth(auxiliary_obs_names_) != auxiliary_state_dim_)
        {
            throw std::runtime_error("VAE observation keys do not match configured dimensions");
        }

        LoadReference();
        inference_counter_ = 0;
        reference_advance_enabled_ = true;
        dance_done_flag = false;
        has_last_motor_targets_ = false;
        last_motor_targets_.setZero();
        for (int i = 0; i < kNumMotors; ++i)
        {
            jpos_des[i] = default_qpos_[i];
        }
        LOG(INFO) << "[VAE] reset motion=" << data_name_ << ", frames=" << end_iter_;
    }

    void FsmVaeController::GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
    {
        (void)gamepad;
        const std::shared_ptr<const MotorState> motor_state =
            robot_interface.motor_state_buffer_.GetData();
        const std::shared_ptr<const ImuState> imu_state =
            robot_interface.imu_state_buffer_.GetData();
        for (int i = 0; i < kNumMotors; ++i)
        {
            joint_pos_[i] = motor_state->jpos[i];
            joint_vel_[i] = motor_state->jvel[i];
        }
        pelvis_gvec_ = RpyToGravity(imu_state->rpy);
        gyro_ = Eigen::Vector3f(
            imu_state->gyro[0], imu_state->gyro[1], imu_state->gyro[2]);

        // This matches the Humanoid_Pipeline environment reset: the first
        // last_motor_targets observation is the measured pose.
        if (!has_last_motor_targets_)
        {
            last_motor_targets_ = joint_pos_;
            has_last_motor_targets_ = true;
        }
    }

    std::vector<float> FsmVaeController::BuildObservation(
        const std::vector<std::string> &names,
        const Eigen::Ref<const Eigen::VectorXf> &ref_qpos,
        const Eigen::Ref<const Eigen::VectorXf> &ref_qvel) const
    {
        std::vector<float> observation;
        observation.reserve(ObservationWidth(names));
        for (const auto &name : names)
        {
            if (name == "gvec_pelvis")
            {
                observation.insert(
                    observation.end(), pelvis_gvec_.data(), pelvis_gvec_.data() + 3);
            }
            else if (name == "gyro_pelvis")
            {
                for (int i = 0; i < 3; ++i)
                {
                    observation.push_back(gyro_[i] * joint_vel_scale_);
                }
            }
            else if (name == "joint_pos")
            {
                for (const int id : obs_joint_ids_)
                {
                    observation.push_back(joint_pos_[id] - default_qpos_[id]);
                }
            }
            else if (name == "joint_vel")
            {
                for (const int id : obs_joint_ids_)
                {
                    observation.push_back(joint_vel_[id] * joint_vel_scale_);
                }
            }
            else if (name == "last_motor_targets")
            {
                for (const int id : obs_joint_ids_)
                {
                    observation.push_back(last_motor_targets_[id]);
                }
            }
            else if (name == "dif_joint_pos")
            {
                for (const int id : obs_joint_ids_)
                {
                    observation.push_back(ref_qpos[id] - joint_pos_[id]);
                }
            }
            else if (name == "dif_joint_vel")
            {
                for (const int id : obs_joint_ids_)
                {
                    observation.push_back(
                        (ref_qvel[id] - joint_vel_[id]) * dif_joint_vel_scale_);
                }
            }
            else
            {
                throw std::runtime_error("unsupported VAE observation key: " + name);
            }
        }

        for (float &value : observation)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("non-finite VAE observation");
            }
            value = std::clamp(value, -100.0f, 100.0f);
        }
        return observation;
    }

    void FsmVaeController::Calculate()
    {
        if (inference_counter_ >= end_iter_)
        {
            dance_done_flag = true;
            return;
        }
        if (!has_last_motor_targets_)
        {
            last_motor_targets_ = joint_pos_;
            has_last_motor_targets_ = true;
        }

        const Eigen::Map<const Eigen::VectorXf> ref_qpos(
            ref_qpos_all_.data() + inference_counter_ * ref_qpos_all_.cols() + 7,
            kNumMotors);
        const Eigen::Map<const Eigen::VectorXf> ref_qvel(
            ref_qvel_all_.data() + inference_counter_ * ref_qvel_all_.cols() + 6,
            kNumMotors);

        std::vector<float> state = BuildObservation(obs_names_, ref_qpos, ref_qvel);
        std::vector<float> auxiliary_state =
            BuildObservation(auxiliary_obs_names_, ref_qpos, ref_qvel);
        std::vector<float> noise(kNoiseDim, 0.0f);
        if (noise_std_ > 0.0f)
        {
            for (float &value : noise)
            {
                value = noise_std_ * normal_distribution_(random_engine_);
            }
        }

        const std::array<int64_t, 2> state_shape = {1, state_dim_};
        const std::array<int64_t, 2> auxiliary_shape = {1, auxiliary_state_dim_};
        const std::array<int64_t, 2> noise_shape = {1, kNoiseDim};
        const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, auxiliary_state.data(), auxiliary_state.size(),
            auxiliary_shape.data(), auxiliary_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, state.data(), state.size(),
            state_shape.data(), state_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, noise.data(), noise.size(),
            noise_shape.data(), noise_shape.size()));

        const std::array<const char *, 3> input_names = {
            "auxiliary_state", "state", "noise"};
        const char *output_name = inference_mode_ == "prior"
            ? "continuous_actions_prior"
            : "continuous_actions";
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
            &output_name, 1);
        const size_t output_size =
            outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (output_size != static_cast<size_t>(action_dim_))
        {
            throw std::runtime_error("VAE action output does not contain 29 values");
        }
        const float *action = outputs[0].GetTensorData<float>();

        std::vector<float> motor_targets = default_qpos_;
        for (int i = 0; i < action_dim_; ++i)
        {
            if (!std::isfinite(action[i]))
            {
                throw std::runtime_error("non-finite VAE action");
            }
            const int id = obs_joint_ids_[i];
            motor_targets[id] = action[i] * action_scale_;
            if (student_use_residual_action_)
            {
                motor_targets[id] += ref_qpos[id];
            }
        }

        last_motor_targets_ = Eigen::Map<const Eigen::VectorXf>(
            motor_targets.data(), motor_targets.size());
        for (int i = 0; i < kNumMotors; ++i)
        {
            jpos_des[i] = motor_targets[i];
        }
        if (reference_advance_enabled_)
        {
            ++inference_counter_;
        }
    }

    std::vector<float> FsmVaeController::GetLog()
    {
        return {};
    }

    Eigen::Vector3f FsmVaeController::RpyToGravity(
        const std::array<float, 3> &rpy) const
    {
        const float roll = rpy[0];
        const float pitch = rpy[1];
        const float yaw = rpy[2];
        const float cy = std::cos(yaw * 0.5f);
        const float sy = std::sin(yaw * 0.5f);
        const float cp = std::cos(pitch * 0.5f);
        const float sp = std::sin(pitch * 0.5f);
        const float cr = std::cos(roll * 0.5f);
        const float sr = std::sin(roll * 0.5f);
        const float w = cr * cp * cy + sr * sp * sy;
        const float x = sr * cp * cy - cr * sp * sy;
        const float y = cr * sp * cy + sr * cp * sy;
        const float z = cr * cp * sy - sr * sp * cy;
        return Eigen::Vector3f(
            -2.0f * (x * z - y * w),
            -2.0f * (y * z + x * w),
            -1.0f + 2.0f * (x * x + y * y));
    }
} // namespace unitree::common
