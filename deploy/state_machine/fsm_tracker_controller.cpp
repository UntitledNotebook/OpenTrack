#include <onnxruntime_cxx_api.h>

#include "fsm_tracker_controller.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    template <typename T, std::size_t N> std::array<T, N> vector2array(const std::vector<T> &vec)
    {
        if (vec.size() != N)
        {
            throw std::runtime_error("Vector size does not match array size.");
        }

        std::array<T, N> arr;
        std::copy_n(vec.begin(), N, arr.begin());
        return arr;
    }

    void FsmTrackerController::LoadParam(fs::path &param_folder)
    {
        // do nothing
    }

    void FsmTrackerController::GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
    {
        const std::shared_ptr<const MotorState> ms = robot_interface.motor_state_buffer_.GetData();
        const std::shared_ptr<const ImuState> imus = robot_interface.imu_state_buffer_.GetData();
        for (int i = 0; i < G1_NUM_MOTOR; ++i)
        {
            joint_pos_[i] = ms->jpos[i]; // 关节角度
            joint_vel_[i] = ms->jvel[i]; // 关节角速度
        }
        pelvis_gvec_ = rpy2rotvec(imus->rpy);
        gyro_gvec_ = Eigen::Vector3f(imus->gyro[0], imus->gyro[1], imus->gyro[2]);
    }

    void FsmTrackerController::Reset()
    {
        YAML::Node config = YAML::LoadFile("../../storage/g1_tracking_constant.yaml");
        auto motor_name_dict = config["motor_names"];
        motor_names_.clear();
        for (const auto &name : motor_name_dict["LEG_L"]) // 左腿关节
            motor_names_.push_back(name.as<std::string>());
        for (const auto &name : motor_name_dict["LEG_R"]) // 右腿关节
            motor_names_.push_back(name.as<std::string>());
        for (const auto &name : motor_name_dict["WAIST"]) // 腰部关节
            motor_names_.push_back(name.as<std::string>());
        for (const auto &name : motor_name_dict["ARM_L"]) // 左臂关节
            motor_names_.push_back(name.as<std::string>());
        for (const auto &name : motor_name_dict["ARM_R"]) // 右臂关节
            motor_names_.push_back(name.as<std::string>());
        kp = ReadYaml(config["KPs"][0]);
        kd = ReadYaml(config["KDs"][0]);
        default_qpos_ = ReadYamlConst(config["DEFAULT_QPOS"]);
        joint_vel_scale_ = config["joint_vel_scale"].as<float>();
        inference_counter_ = 0;
        dance_done_flag = false;
        history_queue_.clear();
        
        // 现在只加载一个onnx模型，路径为"../storage/data/{data_name}/ref_data.onnx"
        std::string ref_data_onnx_path = "../../storage/data/" + data_name_ + "/ref_data.onnx";

        Ort::SessionOptions session_options;
        Ort::AllocatorWithDefaultOptions allocator;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // 加载ref_data.onnx模型
        Ort::Session session_ref_data(env_, ref_data_onnx_path.c_str(), session_options);

        // 需要输出的名称
        std::vector<const char *> output_names = {"qpos", "qvel", "feet_height", "root_height"};

        // 运行模型，不需要输入
        auto outputs = session_ref_data.Run(Ort::RunOptions{nullptr}, nullptr, nullptr, 0, output_names.data(), output_names.size());

        // 处理qpos
        auto tensor_info_qpos = outputs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims_qpos = tensor_info_qpos.GetShape();
        float *data_qpos = outputs[0].GetTensorMutableData<float>();
        ref_qpos_all_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            data_qpos, dims_qpos[0], dims_qpos[1]);

        // 处理qvel
        auto tensor_info_qvel = outputs[1].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims_qvel = tensor_info_qvel.GetShape();
        float *data_qvel = outputs[1].GetTensorMutableData<float>();
        ref_qvel_all_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            data_qvel, dims_qvel[0], dims_qvel[1]);

        // 处理feet_height
        auto tensor_info_feet_height = outputs[2].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims_feet_height = tensor_info_feet_height.GetShape();
        float *data_feet_height = outputs[2].GetTensorMutableData<float>();
        ref_feet_height_all_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            data_feet_height, dims_feet_height[0], dims_feet_height[1]);

        // 处理root_height
        auto tensor_info_root_height = outputs[3].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims_root_height = tensor_info_root_height.GetShape();
        float *data_root_height = outputs[3].GetTensorMutableData<float>();
        ref_root_height_all_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            data_root_height, dims_root_height[0], dims_root_height[1]);

        end_iter_ = ref_qpos_all_.rows();
        VLOG(1) << "end_iter: " << end_iter_;
        obs_joint_ids_ = ReadYamlConstInt(config["OBS_JOINT_IDS"]);
        obs_joint_num_ = obs_joint_ids_.size();

        // 计算obs的长度
        int state_sensor_len = 0;
        for (const auto& name : obs_names_) {
            if (strcmp(name, "gyro_pelvis") == 0) state_sensor_len += 3;
            else if (strcmp(name, "gvec_pelvis") == 0) state_sensor_len += 3;
            else if (strcmp(name, "joint_pos") == 0) state_sensor_len += obs_joint_num_;
            else if (strcmp(name, "joint_vel") == 0) state_sensor_len += obs_joint_num_;
            else if (strcmp(name, "last_motor_targets") == 0) state_sensor_len += obs_joint_num_;
            else if (strcmp(name, "dif_joint_pos") == 0) state_sensor_len += obs_joint_num_;
            else if (strcmp(name, "dif_joint_vel") == 0) state_sensor_len += obs_joint_num_;
            else if (strcmp(name, "ref_feet_height") == 0) state_sensor_len += 4;
            else if (strcmp(name, "ref_root_linvel") == 0 || strcmp(name, "ref_root_linvel_local") == 0) state_sensor_len += 3;
            else if (strcmp(name, "ref_root_angvel") == 0 || strcmp(name, "ref_root_angvel_local") == 0) state_sensor_len += 3;
            else if (strcmp(name, "ref_root_quat") == 0) state_sensor_len += 4;
            else if (strcmp(name, "ref_root_height") == 0) state_sensor_len += 1;
            else throw std::runtime_error(std::string("Unsupported tracker observation key: ") + name);
        }

        VLOG(1) << "input shape: " << state_sensor_len;
        input_shape_ = {1, state_sensor_len};
        VLOG(1) << "history shape: " << (3 + 3 + obs_joint_num_ * 3) * 79;
        history_shape_ = {1, (3 + 3 + obs_joint_num_ * 3), 79};
        last_motor_targets_ = Eigen::VectorXf::Zero(G1_NUM_MOTOR);
    }

    void FsmTrackerController::Calculate() {
        if (inference_counter_ >= end_iter_) // end_iter_
        {
            dance_done_flag = true;
            return;
        }

        // GetInput runs after Reset, so use fresh joint feedback on every entry.
        // Training initializes last_motor_targets to the measured joint positions.
        if (inference_counter_ == 0)
        {
            last_motor_targets_ = joint_pos_;
        }

        // 预先准备所有obs_keys可能需要的数据
        Eigen::Map<Eigen::VectorXf> ref_qpos(ref_qpos_all_.data() + inference_counter_ * ref_qpos_all_.cols() + 7, G1_NUM_MOTOR);
        Eigen::Map<Eigen::VectorXf> ref_qvel(ref_qvel_all_.data() + inference_counter_ * ref_qvel_all_.cols() + 6, G1_NUM_MOTOR);
        Eigen::Map<Eigen::VectorXf> ref_feet_height(ref_feet_height_all_.data() + inference_counter_ * ref_feet_height_all_.cols(), 4);
        Eigen::Map<Eigen::VectorXf> ref_root_height(ref_root_height_all_.data() + inference_counter_ * ref_root_height_all_.cols(), 1);

        Eigen::Map<Eigen::VectorXf> ref_root_quat(ref_qpos_all_.data() + inference_counter_ * ref_qpos_all_.cols() + 3, 4);
        Eigen::Map<Eigen::VectorXf> ref_root_linvel(ref_qvel_all_.data() + inference_counter_ * ref_qvel_all_.cols(), 3);
        Eigen::Map<Eigen::VectorXf> ref_root_angvel(ref_qvel_all_.data() + inference_counter_ * ref_qvel_all_.cols() + 3, 3);

        Eigen::VectorXf dif_qpos = ref_qpos - joint_pos_;
        Eigen::VectorXf dif_qvel = (ref_qvel - joint_vel_) * joint_vel_scale_;

        // 计算局部坐标系下的线速度
        Eigen::Quaternionf q_world_from_local(ref_root_quat(0), ref_root_quat(1), ref_root_quat(2), ref_root_quat(3));
        q_world_from_local.normalize();
        Eigen::Vector3f v_world = ref_root_linvel;
        Eigen::Quaternionf q_local_from_world = q_world_from_local.inverse();
        Eigen::Vector3f ref_root_linvel_local = q_local_from_world * v_world;

        Eigen::VectorXf state_sensor(input_shape_[1]);
        int index = 0;
        for (const auto& name : obs_names_) {
            if (strcmp(name, "gyro_pelvis") == 0) {
                for (int i = 0; i < 3; ++i)
                    state_sensor[index++] = gyro_gvec_[i] * joint_vel_scale_;
            }
            else if (strcmp(name, "gvec_pelvis") == 0) {
                for (int i = 0; i < 3; ++i)
                    state_sensor[index++] = pelvis_gvec_[i];
            }
            else if (strcmp(name, "joint_pos") == 0) {
                for (int id : obs_joint_ids_)
                    state_sensor[index++] = joint_pos_[id] - default_qpos_[id];
            }
            else if (strcmp(name, "joint_vel") == 0) {
                for (int id : obs_joint_ids_)
                    state_sensor[index++] = joint_vel_[id] * joint_vel_scale_;
            }
            else if (strcmp(name, "last_motor_targets") == 0) {
                for (int id : obs_joint_ids_)
                    state_sensor[index++] = last_motor_targets_[id];
            }
            else if (strcmp(name, "dif_joint_pos") == 0) {
                for (int id : obs_joint_ids_)
                    state_sensor[index++] = dif_qpos[id];
            }
            else if (strcmp(name, "dif_joint_vel") == 0) {
                for (int id : obs_joint_ids_)
                    state_sensor[index++] = dif_qvel[id];
            }
            else if (strcmp(name, "ref_feet_height") == 0) {
                for (int i = 0; i < 4; i++)
                    state_sensor[index++] = ref_feet_height[i];
            }
            else if (strcmp(name, "ref_root_linvel") == 0 || strcmp(name, "ref_root_linvel_local") == 0) {
                for (int i = 0; i < 3; i++)
                    state_sensor[index++] = ref_root_linvel_local[i] * joint_vel_scale_;
            }
            else if (strcmp(name, "ref_root_angvel") == 0 || strcmp(name, "ref_root_angvel_local") == 0) {
                for (int i = 0; i < 3; i++)
                    state_sensor[index++] = ref_root_angvel[i] * joint_vel_scale_;
            }
            else if (strcmp(name, "ref_root_quat") == 0) {
                for (int i = 0; i < 4; i++)
                    state_sensor[index++] = ref_root_quat[i];
            }
            else if (strcmp(name, "ref_root_height") == 0) {
                state_sensor[index++] = ref_root_height[0];
            }
            // 可根据需要扩展
        }

        // 裁剪
        for (int i = 0; i < state_sensor.size(); ++i)
        {
            if (state_sensor[i] > 100.0f)
            {
                state_sensor[i] = 100.0f;
            }
            else if (state_sensor[i] < -100.0f)
            {
                state_sensor[i] = -100.0f;
            }
        }
        Eigen::VectorXf nn_obs(state_sensor.size());
        nn_obs = state_sensor.transpose();

        // 如果history_queue_为空，则填上初始值
        if (history_queue_.empty())
        {
            std::vector<float> current_history(3 + 3 + 29 * 3);
            for (int i = 0; i < 3 + 3 + 29 * 2; i++)
            {
                current_history[i] = state_sensor[i];
            }
            for (int i = 0; i < 29; i++)
            {
                current_history[i + 3 + 3 + 29 * 2] = joint_pos_[i];
            }

            for (int i = 0; i < 79; ++i)
            {
                history_queue_.push_back(current_history);
            }
        }
    
        std::vector<float> history_sensor((3 + 3 + 29 * 3) * 79, 0.0f);
        for (int i = 0; i < 3 + 3 + 29 * 3; ++i)
        {
            for (int j = 0; j < 79; ++j)
            {
                history_sensor[i * 79 + j] = history_queue_.at(j).at(i);
            }
        }

        // ONNX输入张量
        std::vector<Ort::Value> input_tensors;
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // 判断input_names_的数量，决定输入obs还是obs+history
        if (input_names_.size() == 1)
        {
            // 只需要obs
            input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, nn_obs.data(), nn_obs.size(),
                                                                    input_shape_.data(), input_shape_.size()));
        }
        else if (input_names_.size() == 2)
        {
            // 需要obs和history
            input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, nn_obs.data(), nn_obs.size(),
                                                                    input_shape_.data(), input_shape_.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, history_sensor.data(), history_sensor.size(),
                                                                    history_shape_.data(), history_shape_.size()));
        }
        else
        {
            LOG(ERROR) << "unsupported input count: " << input_names_.size();
            throw std::runtime_error("unsupported input count");
        }

        auto output_tensors = session_ptr_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                                           input_names_.size(), output_names_.data(), output_names_.size());
        // GetTensorMutableData：获取执行张量数据的原始指针，返回指向张量数据缓冲区起始位置的指针
        float *nn_action_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> nn_action(nn_action_data,
                                     nn_action_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());

        // 计算目标关节位置
        std::vector<float> motor_targets = default_qpos_;
        // TODO: 如果obs和activate
        // actuator的id不一致这里需要分开去配置。目前一致先这样
        for (size_t i = 0; i < obs_joint_ids_.size(); i++)
        {
            int idx = obs_joint_ids_[i];
            motor_targets[idx] = ref_qpos[idx] + nn_action[i] * action_scale_;
        }
        last_motor_targets_ = Eigen::Map<Eigen::VectorXf>(motor_targets.data(), motor_targets.size());

        // 更新history
        std::vector<float> current_history(3 + 3 + 29 * 3);
        for (int i = 0; i < 3 + 3 + 29 * 2; i++)
        {
            current_history[i] = state_sensor[i];
        }
        for (int i = 0; i < 29; i++)
        {
            current_history[i + 3 + 3 + 29 * 2] = motor_targets[i];
        }
        history_queue_.pop_front();
        history_queue_.push_back(current_history);
        

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            jpos_des.at(i) = motor_targets[i];
        }
        inference_counter_++;
    }

    std::vector<float> FsmTrackerController::GetLog()
    {
        std::vector<float> log;
        return log;
    }

} // namespace unitree::common
