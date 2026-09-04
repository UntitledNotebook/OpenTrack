#pragma once

#include "fsm_basic_controller.hpp"
#include <array>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <shared_mutex>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "cfg.hpp"
#include "gamepad.hpp"
#include "robot_interface.hpp"
#include "json.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    class FsmTrackerController : public BasicUserController
    {
    public:
        FsmTrackerController(std::string policy_name, std::string data_name)
                        : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
                            policy_name_(policy_name),
              data_name_(data_name)
        {
            // 构造policy.onnx的完整路径
            std::string checkpoints_dir = "../../storage/policy/" + policy_name + "/checkpoints/";
            int64_t max_number = -1;
            std::string max_dir_name;

            try {
                for (const auto& entry : fs::directory_iterator(checkpoints_dir)) {
                    if (entry.is_directory()) {
                        std::string dirname = entry.path().filename().string();
                        // 检查是否为纯数字
                        bool is_number = !dirname.empty() && std::all_of(dirname.begin(), dirname.end(), ::isdigit);
                        if (is_number) {
                            int64_t num = 0;
                            try {
                                num = std::stoll(dirname); // 使用stoll以支持大数字
                            } catch (const std::exception& e) {
                                LOG(WARNING) << "failed to convert directory name to int64_t: " << dirname << ", error: " << e.what();
                                continue;
                            }
                            if (num > max_number) {
                                max_number = num;
                                max_dir_name = dirname;
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                LOG(ERROR) << "failed to read checkpoints dir: " << checkpoints_dir << ", error: " << e.what();
            }

            // 两种支持的 checkpoint 布局：
            //   1) numeric: checkpoints/<最大数字>/policy.onnx (specialist 现状)
            //   2) flat   : checkpoints/model.onnx              (generalist, e.g. general_tracker_lafan1_v2)
            std::string onnx_path;
            if (max_number != -1) {
                onnx_path = checkpoints_dir + max_dir_name + "/policy.onnx";
                checkpoint_dir_name_ = max_dir_name;
                VLOG(1) << "[CKPT_LAYOUT] numeric, dir=" << max_dir_name;
            } else {
                std::string flat_path = checkpoints_dir + "model.onnx";
                if (fs::exists(flat_path)) {
                    onnx_path = flat_path;
                    checkpoint_dir_name_ = "flat";
                    VLOG(1) << "[CKPT_LAYOUT] flat, file=model.onnx";
                } else {
                    LOG(ERROR) << "checkpoint not found, tried:\n"
                               << "  - " << checkpoints_dir << "<numeric>/policy.onnx\n"
                               << "  - " << flat_path;
                    throw std::runtime_error("checkpoint not found");
                }
            }
            policy_onnx_path_ = onnx_path;
            VLOG(1) << "[tracker] onnx=" << onnx_path;

            session_ptr_ = new Ort::Session(env_, onnx_path.c_str(), session_options_);

            size_t num_input_nodes = session_ptr_->GetInputCount();
            VLOG(1) << "num_input_nodes: " << num_input_nodes;
            for (size_t i = 0; i < num_input_nodes; i++)
            {
                input_names_.push_back(session_ptr_->GetInputNameAllocated(i, allocator_).release());
            }
            size_t num_output_nodes = session_ptr_->GetOutputCount();
            VLOG(1) << "num_output_nodes: " << num_output_nodes;
            for (size_t i = 0; i < num_output_nodes; i++)
            {
                output_names_.push_back(session_ptr_->GetOutputNameAllocated(i, allocator_).release());
            }

            for (auto input_name : input_names_)
            {
                VLOG(1) << "input_name: " << input_name;
            }
            for (auto output_name : output_names_)
            {
                VLOG(1) << "output_name: " << output_name;
            }

            // 读取config.json，获取obs_keys
            try {
                std::string config_json_path = "../../storage/policy/" + policy_name + "/checkpoints/config.json";
                std::ifstream config_ifs(config_json_path);
                if (!config_ifs.is_open()) {
                    LOG(ERROR) << "cannot open config.json: " << config_json_path;
                    throw std::runtime_error("cannot open config.json");
                }
                std::string config_content((std::istreambuf_iterator<char>(config_ifs)), std::istreambuf_iterator<char>());
                config_ifs.close();

                // 解析json
                nlohmann::json config_json = nlohmann::json::parse(config_content);
                if (!config_json.contains("env_config") || !config_json["env_config"].contains("obs_keys")) {
                    LOG(ERROR) << "config.json missing env_config or obs_keys field";
                    throw std::runtime_error("config.json missing env_config or obs_keys field");
                }
                const auto& obs_keys = config_json["env_config"]["obs_keys"];
                obs_names_.clear();
                for (const auto& key : obs_keys) {
                    char* cstr = new char[key.get<std::string>().size() + 1];
                    std::strcpy(cstr, key.get<std::string>().c_str());
                    obs_names_.push_back(cstr);
                }
                VLOG(1) << "obs_names_ count=" << obs_names_.size();
            } catch (const std::exception& e) {
                LOG(ERROR) << "failed to read config.json: " << e.what();
                throw;
            }

            // 其他初始化
            action_scale_ = 1.0f;

            // 初始化需要顺序控制的变量
            joint_pos_ = Eigen::VectorXf::Zero(G1_NUM_MOTOR);
            joint_vel_ = Eigen::VectorXf::Zero(G1_NUM_MOTOR);
        }

        virtual void LoadParam(fs::path &param_folder) override;
        virtual void Reset() override;
        // 读取遥控器控制信息
        virtual void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) override;
        virtual void Calculate() override;
        virtual std::vector<float> GetLog() override;

        const std::string &GetPolicyName() const { return policy_name_; }
        const std::string &GetPolicyCheckpointDir() const { return checkpoint_dir_name_; }
        const std::string &GetPolicyOnnxPath() const { return policy_onnx_path_; }
        const std::string &GetMotionName() const { return data_name_; }

    private:
        static constexpr int G1_NUM_MOTOR = 29;

    private:
        Eigen::Vector3f rpy2rotvec(const std::array<float, 3> &rpy)
        {
            float roll = rpy[0], pitch = rpy[1], yaw = rpy[2];

            // 计算半角三角函数
            float cy = std::cos(yaw * 0.5f);
            float sy = std::sin(yaw * 0.5f);
            float cp = std::cos(pitch * 0.5f);
            float sp = std::sin(pitch * 0.5f);
            float cr = std::cos(roll * 0.5f);
            float sr = std::sin(roll * 0.5f);

            // 通过旋转顺序ZYX计算四元数
            float w = cr * cp * cy + sr * sp * sy;
            float x = sr * cp * cy - cr * sp * sy;
            float y = cr * sp * cy + sr * cp * sy;
            float z = cr * cp * sy - sr * sp * cy;

            // 调用四元数转重力向量
            float gx = -2 * (x * z - y * w);
            float gy = -2 * (y * z + x * w);
            float gz = -1 + 2 * (x * x + y * y);

            return Eigen::Vector3f(gx, gy, gz);
        }

        std::array<float, G1_NUM_MOTOR> ReadYaml(const YAML::Node &node)
        {
            std::array<float, G1_NUM_MOTOR> output;
            if (!node || !node.IsSequence())
            {
                LOG(ERROR) << node << "missing or not a sequence!";
            }
            for (size_t i = 0; i < G1_NUM_MOTOR; ++i)
            {
                output[i] = node[i].as<float>();
            }
            return output;
        }

        const std::vector<float> ReadYamlConst(const YAML::Node &node)
        {
            std::vector<float> output;
            if (!node || !node.IsSequence())
            {
                LOG(ERROR) << node << "missing or not a sequence!";
            }
            output.reserve(node.size());
            for (size_t i = 0; i < node.size(); ++i)
            {
                output.push_back(node[i].as<float>());
            }
            return output;
        }

        const std::vector<int> ReadYamlConstInt(const YAML::Node &node)
        {
            std::vector<int> output;
            if (!node || !node.IsSequence())
            {
                LOG(ERROR) << node << "missing or not a sequence!";
            }
            output.reserve(node.size());
            for (size_t i = 0; i < node.size(); ++i)
            {
                output.push_back(node[i].as<int>());
            }
            return output;
        }

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ReadYamlMat(const YAML::Node &node)
        {
            // 检查节点有效性
            if (!node || !node.IsSequence())
            {
                LOG(ERROR) << "Error: YAML node is missing or not a sequence!";
                return Eigen::MatrixXf::Zero(0, 0); // 返回空矩阵
            }

            const size_t rows = node.size();
            if (rows == 0)
                return Eigen::MatrixXf::Zero(0, 0);

            // 检查第一行确定列数
            const size_t cols = node[0].size();
            for (size_t i = 1; i < rows; ++i)
            {
                if (!node[i].IsSequence() || node[i].size() != cols)
                {
                    LOG(ERROR) << "Error: Inconsistent column size at row " << i << ". Expected " << cols << ", got "
                               << node[i].size();
                    return Eigen::MatrixXf::Zero(0, 0);
                }
            }

            // 创建动态矩阵（行优先）
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(rows, cols);

            // 填充矩阵数据
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    mat(i, j) = node[i][j].as<float>();
                }
            }

            return mat;
        }

    private:
        std::vector<float> default_qpos_;
        std::vector<float> torque_limit_;
        std::vector<int> obs_joint_ids_;
        float joint_vel_scale_;
        int end_iter_;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref_qpos_all_;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref_qvel_all_;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref_feet_height_all_;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref_root_height_all_;
        
        std::vector<std::string> motor_names_;
        int inference_counter_ = 0; // 新增推理计数器
        Eigen::VectorXf last_motor_targets_;
        Ort::Env env_;
        Ort::SessionOptions session_options_;
        Ort::AllocatorWithDefaultOptions allocator_;
        Ort::Session* session_ptr_;
        std::vector<int64_t> input_shape_;
        std::vector<int64_t> history_shape_;
        std::vector<const char *> input_names_;
        std::vector<const char *> output_names_;
        std::vector<const char *> obs_names_;
        float action_scale_;

        Eigen::VectorXf joint_pos_;
        Eigen::VectorXf joint_vel_;
        Eigen::Vector3f pelvis_gvec_;
        Eigen::Vector3f gyro_gvec_;

        std::deque<std::vector<float>> history_queue_;
        std::string policy_name_;
        std::string checkpoint_dir_name_;
        std::string policy_onnx_path_;
        std::string data_name_;
        int obs_joint_num_;
    };
} // namespace unitree::common
