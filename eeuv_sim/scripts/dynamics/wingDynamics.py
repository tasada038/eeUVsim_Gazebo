#!/usr/bin/python3
"""
Script to calculate wing dynamics for glider AUV in Gazebo
@author: Yuya Hamamatsu 
@contact: yuya.hamamatsu@taltech.ee
"""

import math
import numpy as np

class WingDynamics():
    def __init__(self, dynamics_parameters):
        """
            self:
            dynamics_parameters: dictionary variable containg fin dynamics params
        """
        self.c_d = dynamics_parameters["C_D"]
        self.c_l = dynamics_parameters["C_L"]
        self.wing_area = dynamics_parameters["WingArea"]


        alpha_mount = dynamics_parameters["WingMountingAngle"]
        self.alpha_mount_cos = np.cos(np.deg2rad(alpha_mount))
        self.alpha_mount_sin = np.sin(np.deg2rad(alpha_mount))


        self.FxFins = np.zeros(4)
        self.FyFins = np.zeros(4)
        self.FzFins = np.zeros(4)

        self.TxFins = np.zeros(4)
        self.TyFins = np.zeros(4)
        self.TzFins = np.zeros(4)

        
    def _calculate_forces(self, velocity_body, wing_area, angle_of_attack, air_density=998):
        """
        揚力と抗力を計算する関数。
        
        Parameters:
        velocity_body (np.array): 機体座標系での速度ベクトル（m/s）。
        wing_area (float): 翼面積（平方メートル）。
        angle_of_attack (float): 迎角（ラジアン）。
        air_density (float): 空気密度（kg/m^3）。デフォルトは標準海面レベルの1.225 kg/m^3。
        
        Returns:
        (lift, drag): 揚力と抗力のタプル（ニュートン単位）。
        """
        # 速度の大きさ
        speed = np.linalg.norm(velocity_body)
        
        cl_0 = 0.0  # 迎え角が0のときの揚力係数
        cd_0 = self.c_d[1]  # 基本抗力係数
        
        # 揚力係数（Cl）を計算
        cl = self.c_l[0] * angle_of_attack + self.c_l[0] * angle_of_attack * abs(angle_of_attack) + cl_0
        
        # 抗力係数（Cd）を計算
        cd = self.c_d[0] * angle_of_attack * angle_of_attack + cd_0
        
        # 動圧（q）を計算
        q = 0.5 * air_density * (speed ** 2)
        
        # 揚力（L）を計算
        lift = cl * q * wing_area
        
        # 抗力（D）を計算
        drag = cd * q * wing_area
        
        return lift, drag

    def calculate_wrench(self, velocity_world, wing_area, distance_to_cg, body_orientation, default_wing_angle, control_angle, axis='pitch', air_density=998):
        """
        翼の迎え角、速度、翼面積に基づいて機体に加わる力とモーメント（レンチ）を計算する関数。
        
        Parameters:
        velocity_world (np.array): 世界座標系での速度（m/s）。
        wing_area (float): 翼面積（平方メートル）。
        distance_to_cg (np.array): 翼の揚力中心から機体の重心までの距離（x, y, z）メートル単位。
        body_orientation (tuple): 世界座標系に対する機体の姿勢（ロール、ピッチ、ヨー）度単位。
        default_wing_angle (tuple): 翼のデフォルトの取り付け角度（ロール、ピッチ、ヨー）度単位。
        control_angle (float): 操作角度（度単位）。
        axis (str): 'pitch' または 'yaw'。操作軸を指定。
        air_density (float): 空気密度（kg/m^3）。デフォルトは標準海面レベルの1.225 kg/m^3。
        
        Returns:
        wrench: 機体に加わる力とモーメントのタプル（ニュートン・ニュートンメートル単位）。
        """
        # 速度を機体座標系に変換する
        rotation_matrix_world_to_body = self._euler_to_rotation_matrix(*body_orientation).T
        velocity_body = np.dot(rotation_matrix_world_to_body, velocity_world)
        
        # 操作角度をラジアンに変換
        control_angle_rad = math.radians(control_angle)
        
        # デフォルトの取り付け角度をラジアンに変換
        default_wing_angle_rad = np.radians(default_wing_angle)
        
        # 翼の迎角を計算
        if np.linalg.norm(velocity_body) > 0:
            if axis == 'pitch':
                angle_of_attack = math.atan2(velocity_body[2], velocity_body[0]) + default_wing_angle_rad + control_angle_rad
            elif axis == 'roll':
                angle_of_attack = math.atan2(velocity_body[2], velocity_body[1]) + default_wing_angle_rad + control_angle_rad
            elif axis == 'yaw':
                angle_of_attack = math.atan2(velocity_body[1], velocity_body[0]) + default_wing_angle_rad + control_angle_rad
            else:
                raise ValueError("Axis must be 'pitch' or 'yaw'")
        else:
            angle_of_attack = default_wing_angle_rad + control_angle_rad
        
        # 翼の揚力と抗力を計算
        lift, drag = self._calculate_forces(velocity_body, wing_area, angle_of_attack, air_density)
        
        # 速度方向ベクトルを正規化
        if np.linalg.norm(velocity_body) > 0:
            velocity_dir_body = velocity_body / np.linalg.norm(velocity_body)
        else:
            velocity_dir_body = np.array([0, 0, 0])
        
        # 揚力の方向を決定
        if np.linalg.norm(velocity_body) > 0:
            if axis == 'pitch':
                lift_direction_body = np.array([-velocity_dir_body[2], 0, velocity_dir_body[0]])
            elif axis == 'roll':
                lift_direction_body = np.array([0, -velocity_dir_body[2], velocity_dir_body[1]])
            elif axis == 'yaw':
                lift_direction_body = np.array([-velocity_dir_body[1], velocity_dir_body[0], 0])
            lift_direction_body /= np.linalg.norm(lift_direction_body)
        else:
            lift_direction_body = np.array([0, 0, 0])

        # print("------------------------------")
        # print("DYANAMICS")
        # print("lift_direction_body", lift_direction_body)
        # print("velocity_dir_body", velocity_dir_body)
        # print("distance_to_cg", distance_to_cg)
        # print("lift", lift)
        # print("drag", drag)
        # print("body_orientation", body_orientation)
        # print("rotation_matrix_world_to_body", rotation_matrix_world_to_body)
        # print("velocity_world", velocity_world)
        # print("velocity_body", velocity_body)
        # print("--------------------------------")

        
        # 翼の力を合成
        lift_vector = lift * lift_direction_body
        drag_vector = -drag * velocity_dir_body
        force_body = lift_vector + drag_vector
        
        # 力のモーメントを計算
        moment_body = np.cross(distance_to_cg, force_body)
        
        # 力とモーメントを世界座標系に変換
        rotation_matrix_body_to_world = self._euler_to_rotation_matrix(*body_orientation)
        force_world = np.dot(rotation_matrix_body_to_world, force_body)
        moment_world = np.dot(rotation_matrix_body_to_world, moment_body)
        
        # return 6-DOF wrench list
        wrench = np.concatenate((force_world, moment_world))

        return wrench

    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        オイラー角を回転行列に変換する関数。
        
        Parameters:
        roll (float): ロール角（度単位）。
        pitch (float): ピッチ角（度単位）。
        yaw (float): ヨー角（度単位）。
        
        Returns:
        rotation_matrix: 回転行列（3x3のnumpy配列）。
        """
        
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        
        Ry = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        Rz = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
        
        return rotation_matrix

