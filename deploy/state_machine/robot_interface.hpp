#pragma once

#include <array>
#include <iostream>
#include <shared_mutex>
#include <glog/logging.h>
#include "comm.h"
#include "unitree/idl/hg/LowState_.hpp"
#include "unitree/idl/hg/LowCmd_.hpp"
#include "conversion.hpp"

namespace unitree::common
{

    constexpr double PosStopF = (2.146E+9f);
    constexpr double VelStopF = (16000.0f);
    struct MotorCommand
    {
        MotorCommand()
        {
            jpos_des.fill(0.0);
            jvel_des.fill(0.0);
            kp.fill(0.0);
            kd.fill(0.0);
            tau_ff.fill(0.0);
        }
        std::array<float, 29> jpos_des, jvel_des, kp, kd, tau_ff;
        uint8_t mode_machine = 0;
    };
    struct MotorState
    {
        MotorState()
        {
            jpos.fill(0.0);
            jvel.fill(0.0);
            tau.fill(0.0);
        }
        std::array<float, 29> jpos, jvel, tau;
        uint8_t mode_machine = 0;
    };
    struct ImuState
    {
        ImuState()
        {
            quat.fill(0.0);
            rpy.fill(0.0);
            gyro.fill(0.0);
            projected_gravity.fill(0.0);
        }
        std::array<float, 4> quat;
        std::array<float, 3> rpy, gyro, projected_gravity;
    };

    template <typename T>
    class DataBuffer
    {
    public:
        void SetData(const T &newData)
        {
            std::unique_lock<std::shared_mutex> lock(mutex);
            data = std::make_shared<T>(newData);
        }

        std::shared_ptr<const T> GetData()
        {
            std::shared_lock<std::shared_mutex> lock(mutex);
            return data ? data : nullptr;
        }

        void Clear()
        {
            std::unique_lock<std::shared_mutex> lock(mutex);
            data = nullptr;
        }

    private:
        std::shared_ptr<T> data;
        std::shared_mutex mutex;
    };

    class BasicRobotInterface
    {
    public:
        BasicRobotInterface()
        {
        }

        // 把订阅到的数据，写入quat、rpygyro，jpos，jvel， tau
        void LoadState(unitree_hg::msg::dds_::LowState_ &state)
        {
            if (state.crc() != crc32_core((uint32_t *)&state, (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1))
            {
                LOG(ERROR) << "[ERROR] CRC Error";
                return;
            }
            // imu
            const unitree_hg::msg::dds_::IMUState_ &imu = state.imu_state();

            ImuState imu_state;
            imu_state.quat = imu.quaternion();
            imu_state.rpy = imu.rpy();
            imu_state.gyro = imu.gyroscope();
            // inverse quat
            float w = imu_state.quat.at(0);
            float x = -imu_state.quat.at(1);
            float y = -imu_state.quat.at(2);
            float z = -imu_state.quat.at(3);

            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;
            float w2 = w * w;
            float xy = x * y;
            float xz = x * z;
            float yz = y * z;
            float wx = w * x;
            float wy = w * y;
            float wz = w * z;

            imu_state.projected_gravity.at(0) = -2 * (xz + wy);
            imu_state.projected_gravity.at(1) = -2 * (yz - wx);
            imu_state.projected_gravity.at(2) = -(w2 - x2 - y2 + z2);

            imu_state_buffer_.SetData(imu_state);

            // motor
            MotorState motor_state;
            for (size_t i = 0; i < 29; ++i)
            {
                motor_state.jpos.at(i) = state.motor_state()[i].q();
                motor_state.jvel.at(i) = state.motor_state()[i].dq();
                motor_state.tau.at(i) = state.motor_state()[i].tau_est();
            }

            motor_state.mode_machine = state.mode_machine();
            motor_state_buffer_.SetData(motor_state);
        }

        virtual bool GetLowCmd(unitree_hg::msg::dds_::LowCmd_ &cmd) = 0;

        DataBuffer<MotorState> motor_state_buffer_;
        DataBuffer<MotorCommand> motor_command_buffer_;
        DataBuffer<ImuState> imu_state_buffer_;
    };

    class RobotInterface : public BasicRobotInterface
    {
    public:
        RobotInterface() : BasicRobotInterface()
        {
            InitLowCmd();
        }

        bool GetLowCmd(unitree_hg::msg::dds_::LowCmd_ &cmd)
        {
            const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
            if (mc == nullptr)
            {
                return false;
            }
            for (int i = 0; i < 29; ++i)
            {
                low_cmd.motor_cmd()[i].q() = mc->jpos_des.at(i);
                low_cmd.motor_cmd()[i].dq() = mc->jvel_des.at(i);
                low_cmd.motor_cmd()[i].kp() = mc->kp.at(i);
                low_cmd.motor_cmd()[i].kd() = mc->kd.at(i);
                low_cmd.motor_cmd()[i].tau() = mc->tau_ff.at(i);
            }

            low_cmd.mode_pr() = 0;
            low_cmd.mode_machine() = mc->mode_machine;
            // The final 32-bit field stores the checksum from the previous
            // command because low_cmd is reused. Unitree's CRC span excludes
            // that field (word_count - 1), but clear it explicitly before
            // every calculation so the checksum bytes are in their neutral
            // state while a frame is assembled.
            low_cmd.crc() = 0U;
            low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(low_cmd) >> 2) - 1);

            cmd = low_cmd;
            return true;
        }

    private:
        void InitLowCmd()
        {
            low_cmd.crc() = 0U;
            for (int i = 0; i < 29; i++)
            {
                low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
                low_cmd.motor_cmd()[i].q() = (PosStopF);
                low_cmd.motor_cmd()[i].kp() = (0);
                low_cmd.motor_cmd()[i].dq() = (VelStopF);
                low_cmd.motor_cmd()[i].kd() = (0);
                low_cmd.motor_cmd()[i].tau() = (0);
            }
        }

    private:
        unitree_hg::msg::dds_::LowCmd_ low_cmd;
    };
} // namespace unitree::common
