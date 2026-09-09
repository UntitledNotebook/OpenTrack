#include "fsm_scaletrack_controller.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <set>
#include <stdexcept>

#include "json.hpp"

namespace unitree::common
{
    namespace
    {
        constexpr std::array<const char *, 29> kJointNames = {{
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        }};

        fs::path ResolveLatestCheckpoint(const std::string &policy_name, std::string &directory_name)
        {
            const fs::path checkpoints = fs::path("../../storage/policy") / policy_name / "checkpoints";
            int64_t latest = -1;
            fs::path latest_path;
            if (!fs::is_directory(checkpoints))
            {
                throw std::runtime_error("ScaleTrack checkpoints directory not found: " + checkpoints.string());
            }
            for (const auto &entry : fs::directory_iterator(checkpoints))
            {
                if (!entry.is_directory())
                {
                    continue;
                }
                const std::string name = entry.path().filename().string();
                if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char value) {
                        return std::isdigit(value) != 0;
                    }))
                {
                    continue;
                }
                try
                {
                    const int64_t iteration = std::stoll(name);
                    const fs::path model = entry.path() / "policy.onnx";
                    const fs::path metadata = entry.path() / "metadata.json";
                    if (iteration > latest && fs::is_regular_file(model) && fs::is_regular_file(metadata))
                    {
                        latest = iteration;
                        latest_path = entry.path();
                        directory_name = name;
                    }
                }
                catch (const std::exception &)
                {
                    continue;
                }
            }
            if (latest < 0)
            {
                throw std::runtime_error(
                    "no numeric ScaleTrack checkpoint containing policy.onnx and metadata.json under " +
                    checkpoints.string());
            }
            return latest_path;
        }

        template <size_t N>
        std::array<float, N> JsonFloatArray(const nlohmann::json &metadata, const char *key)
        {
            if (!metadata.contains(key) || !metadata[key].is_array() || metadata[key].size() != N)
            {
                throw std::runtime_error(
                    std::string("ScaleTrack metadata field ") + key + " must contain " +
                    std::to_string(N) + " values");
            }
            std::array<float, N> result{};
            for (size_t index = 0; index < N; ++index)
            {
                result[index] = metadata[key][index].get<float>();
                if (!std::isfinite(result[index]))
                {
                    throw std::runtime_error(std::string("non-finite ScaleTrack metadata field: ") + key);
                }
            }
            return result;
        }

        void ShiftAndAppend(std::vector<float> &history, const float *values, size_t width)
        {
            std::rotate(history.begin(), history.begin() + static_cast<std::ptrdiff_t>(width), history.end());
            std::copy_n(values, width, history.end() - static_cast<std::ptrdiff_t>(width));
        }
    } // namespace

    FsmScaleTrackController::FsmScaleTrackController(
        std::string policy_name, std::string data_name)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ScaleTrackONNXRuntime"),
          policy_name_(std::move(policy_name)),
          data_name_(std::move(data_name))
    {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const fs::path checkpoint = ResolveLatestCheckpoint(policy_name_, checkpoint_dir_name_);
        const fs::path policy_path = checkpoint / "policy.onnx";
        const fs::path metadata_path = checkpoint / "metadata.json";
        policy_onnx_path_ = policy_path.string();
        LoadPolicyMetadata(metadata_path);
        policy_session_ = std::make_unique<Ort::Session>(
            env_, policy_onnx_path_.c_str(), session_options_);
        ValidatePolicyInterface();
        LoadReference();

        root_quat_history_.resize(kContext * 4);
        base_ang_vel_history_.resize(kContext * 3);
        dof_pos_history_.resize(kContext * kNumMotors);
        dof_vel_history_.resize(kContext * kNumMotors);
        action_history_.resize(kContext * kNumMotors);
        future_body_positions_base_.resize(kFuture * kSelectedBodies * 3);
        future_body_quaternions_base_.resize(kFuture * kSelectedBodies * 4);

        LOG(INFO) << "[SCALETRACK] policy=" << policy_onnx_path_
                  << ", motion=" << data_name_
                  << ", frames=" << reference_frames_
                  << ", mode=" << mode_index_;
    }

    void FsmScaleTrackController::LoadParam(fs::path &param_folder)
    {
        (void)param_folder;
    }

    void FsmScaleTrackController::LoadPolicyMetadata(const fs::path &metadata_path)
    {
        std::ifstream stream(metadata_path);
        if (!stream)
        {
            throw std::runtime_error("cannot open ScaleTrack metadata: " + metadata_path.string());
        }
        nlohmann::json metadata;
        stream >> metadata;
        if (metadata.value("format", std::string()) != "opentrack_scaletrack_policy_v1")
        {
            throw std::runtime_error("unsupported ScaleTrack metadata format");
        }
        if (metadata.value("history_buffer_size", 0) != kContext)
        {
            throw std::runtime_error("ScaleTrack policy history must be 3 frames");
        }
        if (std::fabs(metadata.value("control_frequency_hz", 0.0f) - 50.0f) > 1.0e-4f)
        {
            throw std::runtime_error("ScaleTrack policy control frequency must be 50 Hz");
        }
        if (!metadata.contains("joint_names") || metadata["joint_names"].size() != kNumMotors)
        {
            throw std::runtime_error("ScaleTrack metadata must contain 29 joint names");
        }
        for (int index = 0; index < kNumMotors; ++index)
        {
            if (metadata["joint_names"][index].get<std::string>() != kJointNames[index])
            {
                throw std::runtime_error(
                    "ScaleTrack/OpenTrack joint order mismatch at index " + std::to_string(index));
            }
        }
        if (!metadata.contains("selected_body_names") ||
            metadata["selected_body_names"].size() != kSelectedBodies)
        {
            throw std::runtime_error("ScaleTrack metadata must contain 14 selected bodies");
        }
        if (!metadata.contains("future_idx") || metadata["future_idx"].size() != kFuture)
        {
            throw std::runtime_error("ScaleTrack metadata must contain 6 future offsets");
        }
        for (int index = 0; index < kFuture; ++index)
        {
            future_offsets_[index] = metadata["future_idx"][index].get<int64_t>();
            if (future_offsets_[index] < 0)
            {
                throw std::runtime_error("deployment future offsets must be non-negative");
            }
            time_offsets_input_[index] = future_offsets_[index];
        }

        default_dof_pos_ = JsonFloatArray<kNumMotors>(metadata, "default_dof_pos");
        kp = JsonFloatArray<kNumMotors>(metadata, "stiffness");
        kd = JsonFloatArray<kNumMotors>(metadata, "damping");
        init_pos = default_dof_pos_;

        mode_index_ = metadata.value("default_mode_index", 7);
        if (const char *mode = std::getenv("G1_SCALETRACK_MODE"))
        {
            char *end = nullptr;
            const long parsed = std::strtol(mode, &end, 10);
            if (end == mode || *end != '\0' || parsed < 0 || parsed > 7)
            {
                throw std::runtime_error("G1_SCALETRACK_MODE must be an integer in [0, 7]");
            }
            mode_index_ = static_cast<int>(parsed);
        }
        if (mode_index_ < 0 || mode_index_ > 7)
        {
            throw std::runtime_error("ScaleTrack mode index is outside [0, 7]");
        }
        mode_index_input_[0] = mode_index_;
    }

    void FsmScaleTrackController::ValidatePolicyInterface()
    {
        const std::array<const char *, 9> expected_inputs = {{
            "root_quat_buffer", "base_ang_vel_buffer", "dof_pos_buffer",
            "dof_vel_buffer", "last_action_buffer",
            "target_body_pos_future_to_robot_base",
            "target_body_rot_future_to_robot_base", "mode_index", "time_offsets",
        }};
        const std::array<const char *, 2> expected_outputs = {{"motor_targets", "action"}};
        if (policy_session_->GetInputCount() != expected_inputs.size() ||
            policy_session_->GetOutputCount() != expected_outputs.size())
        {
            throw std::runtime_error("ScaleTrack ONNX must expose 9 inputs and 2 outputs");
        }
        for (size_t index = 0; index < expected_inputs.size(); ++index)
        {
            const auto name = policy_session_->GetInputNameAllocated(index, allocator_);
            if (std::string(name.get()) != expected_inputs[index])
            {
                throw std::runtime_error("unexpected ScaleTrack ONNX input order/name");
            }
        }
        for (size_t index = 0; index < expected_outputs.size(); ++index)
        {
            const auto name = policy_session_->GetOutputNameAllocated(index, allocator_);
            if (std::string(name.get()) != expected_outputs[index])
            {
                throw std::runtime_error("unexpected ScaleTrack ONNX output order/name");
            }
        }
    }

    void FsmScaleTrackController::LoadReference()
    {
        const fs::path reference_path =
            fs::path("../../storage/data") / data_name_ / "scaletrack_ref.onnx";
        if (!fs::is_regular_file(reference_path))
        {
            throw std::runtime_error("ScaleTrack reference not found: " + reference_path.string());
        }
        Ort::Session reference_session(env_, reference_path.string().c_str(), session_options_);
        const std::array<const char *, 2> output_names = {{"body_pos_w", "body_quat_w"}};
        auto outputs = reference_session.Run(
            Ort::RunOptions{nullptr}, nullptr, nullptr, 0,
            output_names.data(), output_names.size());

        const auto pos_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto quat_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (pos_shape.size() != 3 || pos_shape[0] <= 0 ||
            pos_shape[1] != kSelectedBodies || pos_shape[2] != 3)
        {
            throw std::runtime_error("body_pos_w must have shape [N,14,3]");
        }
        if (quat_shape.size() != 3 || quat_shape[0] != pos_shape[0] ||
            quat_shape[1] != kSelectedBodies || quat_shape[2] != 4)
        {
            throw std::runtime_error("body_quat_w must have shape [N,14,4]");
        }
        reference_frames_ = static_cast<int>(pos_shape[0]);
        const size_t pos_count = static_cast<size_t>(reference_frames_) * kSelectedBodies * 3;
        const size_t quat_count = static_cast<size_t>(reference_frames_) * kSelectedBodies * 4;
        const float *positions = outputs[0].GetTensorData<float>();
        const float *quaternions = outputs[1].GetTensorData<float>();
        reference_body_positions_.assign(positions, positions + pos_count);
        reference_body_quaternions_.assign(quaternions, quaternions + quat_count);
    }

    void FsmScaleTrackController::Reset()
    {
        inference_counter_ = 0;
        dance_done_flag = false;
        reference_advance_enabled_ = true;
        state_history_initialized_ = false;
        world_alignment_initialized_ = false;
        world_alignment_ = Eigen::Quaternionf::Identity();
        std::fill(root_quat_history_.begin(), root_quat_history_.end(), 0.0f);
        std::fill(base_ang_vel_history_.begin(), base_ang_vel_history_.end(), 0.0f);
        std::fill(dof_pos_history_.begin(), dof_pos_history_.end(), 0.0f);
        std::fill(dof_vel_history_.begin(), dof_vel_history_.end(), 0.0f);
        std::fill(action_history_.begin(), action_history_.end(), 0.0f);
        previous_action_.fill(0.0f);
        for (int index = 0; index < kNumMotors; ++index)
        {
            jpos_des[index] = default_dof_pos_[index];
        }
        LOG(INFO) << "[SCALETRACK] reset motion=" << data_name_
                  << ", frames=" << reference_frames_
                  << ", mode=" << mode_index_;
    }

    void FsmScaleTrackController::GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
    {
        (void)gamepad;
        const auto motor_state = robot_interface.motor_state_buffer_.GetData();
        const auto imu_state = robot_interface.imu_state_buffer_.GetData();
        if (motor_state == nullptr || imu_state == nullptr)
        {
            throw std::runtime_error("ScaleTrack controller has no robot state");
        }
        for (int index = 0; index < kNumMotors; ++index)
        {
            joint_pos_[index] = motor_state->jpos[index];
            joint_vel_[index] = motor_state->jvel[index];
        }
        root_quat_ = imu_state->quat;
        base_ang_vel_ = imu_state->gyro;
        const Eigen::Quaternionf normalized = NormalizeQuaternion(root_quat_.data(), "robot root quaternion");
        root_quat_ = {{normalized.w(), normalized.x(), normalized.y(), normalized.z()}};
        PushStateHistory();
    }

    void FsmScaleTrackController::PushStateHistory()
    {
        if (!state_history_initialized_)
        {
            for (int frame = 0; frame < kContext; ++frame)
            {
                std::copy(root_quat_.begin(), root_quat_.end(), root_quat_history_.begin() + frame * 4);
                std::copy(base_ang_vel_.begin(), base_ang_vel_.end(), base_ang_vel_history_.begin() + frame * 3);
                std::copy(joint_pos_.begin(), joint_pos_.end(), dof_pos_history_.begin() + frame * kNumMotors);
                std::copy(joint_vel_.begin(), joint_vel_.end(), dof_vel_history_.begin() + frame * kNumMotors);
            }
            state_history_initialized_ = true;
            return;
        }
        ShiftAndAppend(root_quat_history_, root_quat_.data(), 4);
        ShiftAndAppend(base_ang_vel_history_, base_ang_vel_.data(), 3);
        ShiftAndAppend(dof_pos_history_, joint_pos_.data(), kNumMotors);
        ShiftAndAppend(dof_vel_history_, joint_vel_.data(), kNumMotors);
    }

    void FsmScaleTrackController::PushActionHistory()
    {
        ShiftAndAppend(action_history_, previous_action_.data(), kNumMotors);
    }

    Eigen::Quaternionf FsmScaleTrackController::NormalizeQuaternion(
        const float *values, const char *name) const
    {
        Eigen::Quaternionf quaternion(values[0], values[1], values[2], values[3]);
        const float norm = quaternion.norm();
        if (!std::isfinite(norm) || norm < 1.0e-6f)
        {
            throw std::runtime_error(std::string("invalid ") + name);
        }
        quaternion.coeffs() /= norm;
        return quaternion;
    }

    Eigen::Quaternionf FsmScaleTrackController::ReferenceQuaternion(int frame, int body) const
    {
        const size_t offset = (static_cast<size_t>(frame) * kSelectedBodies + body) * 4;
        return NormalizeQuaternion(reference_body_quaternions_.data() + offset, "reference quaternion");
    }

    Eigen::Vector3f FsmScaleTrackController::ReferencePosition(int frame, int body) const
    {
        const size_t offset = (static_cast<size_t>(frame) * kSelectedBodies + body) * 3;
        return Eigen::Vector3f(
            reference_body_positions_[offset],
            reference_body_positions_[offset + 1],
            reference_body_positions_[offset + 2]);
    }

    Eigen::Quaternionf FsmScaleTrackController::YawOnly(const Eigen::Quaternionf &quaternion)
    {
        const float sin_yaw = 2.0f * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y());
        const float cos_yaw = 1.0f - 2.0f * (quaternion.y() * quaternion.y() + quaternion.z() * quaternion.z());
        return Eigen::Quaternionf(Eigen::AngleAxisf(std::atan2(sin_yaw, cos_yaw), Eigen::Vector3f::UnitZ()));
    }

    void FsmScaleTrackController::BuildFutureTargets()
    {
        const Eigen::Quaternionf robot_quaternion =
            NormalizeQuaternion(root_quat_.data(), "robot root quaternion");
        if (!world_alignment_initialized_)
        {
            world_alignment_ = YawOnly(robot_quaternion) * YawOnly(ReferenceQuaternion(0, 0)).conjugate();
            world_alignment_.normalize();
            world_alignment_initialized_ = true;
        }

        const Eigen::Vector3f reference_root = ReferencePosition(inference_counter_, 0);
        for (int future = 0; future < kFuture; ++future)
        {
            const int frame = std::min(
                reference_frames_ - 1,
                inference_counter_ + static_cast<int>(future_offsets_[future]));
            for (int body = 0; body < kSelectedBodies; ++body)
            {
                // No global base translation is available on the real robot.
                // Match ScaleTrack local_tracking semantics: use the current
                // reference root translation as the virtual robot position,
                // while retaining measured orientation and initial yaw alignment.
                const Eigen::Vector3f world_delta =
                    world_alignment_ * (ReferencePosition(frame, body) - reference_root);
                const Eigen::Vector3f base_position = robot_quaternion.conjugate() * world_delta;
                const size_t pos_offset = (future * kSelectedBodies + body) * 3;
                future_body_positions_base_[pos_offset] = base_position.x();
                future_body_positions_base_[pos_offset + 1] = base_position.y();
                future_body_positions_base_[pos_offset + 2] = base_position.z();

                Eigen::Quaternionf base_rotation =
                    robot_quaternion.conjugate() * world_alignment_ * ReferenceQuaternion(frame, body);
                base_rotation.normalize();
                const size_t quat_offset = (future * kSelectedBodies + body) * 4;
                future_body_quaternions_base_[quat_offset] = base_rotation.w();
                future_body_quaternions_base_[quat_offset + 1] = base_rotation.x();
                future_body_quaternions_base_[quat_offset + 2] = base_rotation.y();
                future_body_quaternions_base_[quat_offset + 3] = base_rotation.z();
            }
        }
    }

    void FsmScaleTrackController::Calculate()
    {
        if (inference_counter_ >= reference_frames_)
        {
            dance_done_flag = true;
            return;
        }
        if (!state_history_initialized_)
        {
            throw std::runtime_error("ScaleTrack Calculate called before GetInput");
        }
        PushActionHistory();
        BuildFutureTargets();

        const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> root_shape = {{1, kContext, 4}};
        const std::array<int64_t, 3> gyro_shape = {{1, kContext, 3}};
        const std::array<int64_t, 3> joint_shape = {{1, kContext, kNumMotors}};
        const std::array<int64_t, 4> future_pos_shape = {{1, kFuture, kSelectedBodies, 3}};
        const std::array<int64_t, 4> future_quat_shape = {{1, kFuture, kSelectedBodies, 4}};
        const std::array<int64_t, 1> mode_shape = {{1}};
        const std::array<int64_t, 3> time_shape = {{1, kFuture, 1}};

        std::vector<Ort::Value> inputs;
        inputs.reserve(9);
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, root_quat_history_.data(), root_quat_history_.size(), root_shape.data(), root_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, base_ang_vel_history_.data(), base_ang_vel_history_.size(), gyro_shape.data(), gyro_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, dof_pos_history_.data(), dof_pos_history_.size(), joint_shape.data(), joint_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, dof_vel_history_.data(), dof_vel_history_.size(), joint_shape.data(), joint_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, action_history_.data(), action_history_.size(), joint_shape.data(), joint_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, future_body_positions_base_.data(), future_body_positions_base_.size(),
            future_pos_shape.data(), future_pos_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, future_body_quaternions_base_.data(), future_body_quaternions_base_.size(),
            future_quat_shape.data(), future_quat_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory, mode_index_input_.data(), mode_index_input_.size(), mode_shape.data(), mode_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory, time_offsets_input_.data(), time_offsets_input_.size(), time_shape.data(), time_shape.size()));

        const std::array<const char *, 9> input_names = {{
            "root_quat_buffer", "base_ang_vel_buffer", "dof_pos_buffer",
            "dof_vel_buffer", "last_action_buffer",
            "target_body_pos_future_to_robot_base",
            "target_body_rot_future_to_robot_base", "mode_index", "time_offsets",
        }};
        const std::array<const char *, 2> output_names = {{"motor_targets", "action"}};
        auto outputs = policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
            output_names.data(), output_names.size());
        if (outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != kNumMotors ||
            outputs[1].GetTensorTypeAndShapeInfo().GetElementCount() != kNumMotors)
        {
            throw std::runtime_error("ScaleTrack ONNX outputs must contain 29 values each");
        }
        const float *targets = outputs[0].GetTensorData<float>();
        const float *actions = outputs[1].GetTensorData<float>();
        for (int index = 0; index < kNumMotors; ++index)
        {
            if (!std::isfinite(targets[index]) || !std::isfinite(actions[index]))
            {
                throw std::runtime_error("ScaleTrack policy produced NaN or Inf");
            }
            jpos_des[index] = targets[index];
            previous_action_[index] = actions[index];
        }
        if (reference_advance_enabled_)
        {
            ++inference_counter_;
        }
    }

    std::vector<float> FsmScaleTrackController::GetLog()
    {
        return {};
    }
} // namespace unitree::common
