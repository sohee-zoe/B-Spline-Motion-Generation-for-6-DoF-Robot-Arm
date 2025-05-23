import math
import quaternionic
import numpy as np
from numpy import deg2rad as rad
from numpy import rad2deg as deg
from quaternionic import converters
from dual_quaternions import DualQuaternion


def parse_pose(file_name=None):
    if not file_name:
        return
    poses = []
    with open(file_name, "r") as file:
        for line in file:
            int_pose = [float(val) for val in line.split(",")]
            poses.append(int_pose)
            # print(poses)
            # continue
    return poses


def to_quaternion(roll, pitch, yaw):
    # Abbreviations for the various angular functions
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def to_euler_angles(q):
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q[0] * q[2] - q[1] * q[3])
    pitch = math.asin(max(-1, min(1, sinp)))  # Clamped to handle numerical issues

    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [deg(roll), deg(pitch), deg(yaw)]


def get_translation(dq):
    quat = (dq.q_d * dq.q_r.conjugate() - dq.q_r * dq.q_d.conjugate()) / (
        dq.q_r * dq.q_r.conjugate()
    )
    return [quat.x, quat.y, quat.z]


def draw_axis(position, axis, np, passed=0, full=True):
    origin = np.matrix(
        [[position[0] / 1000], [position[1] / 1000], [position[2] / 1000]]
    )
    x, y, z = (
        math.radians(position[3]),
        math.radians(position[4]),
        math.radians(position[5]),
    )
    rotation_matrix = np.matrix(
        [
            [
                math.cos(z) * math.cos(y),
                math.sin(x) * math.sin(y) * math.cos(z) - math.cos(x) * math.sin(z),
                math.cos(x) * math.sin(y) * math.cos(z) + math.sin(x) * math.sin(z),
            ],
            [
                math.sin(z) * math.cos(y),
                math.sin(x) * math.sin(y) * math.sin(z) + math.cos(x) * math.cos(z),
                math.cos(x) * math.sin(y) * math.sin(z) - math.sin(x) * math.cos(z),
            ],
            [-math.sin(y), math.sin(x) * math.cos(y), math.cos(x) * math.cos(y)],
        ]
    )
    x_arr = np.add(np.matmul(rotation_matrix, np.matrix([[0.05], [0], [0]])), origin)
    y_arr = np.add(np.matmul(rotation_matrix, np.matrix([[0], [0.05], [0]])), origin)
    z_arr = np.add(np.matmul(rotation_matrix, np.matrix([[0], [0], [0.05]])), origin)
    if full:
        axis.plot(
            [origin[0, 0], x_arr[0, 0]],
            [origin[1, 0], x_arr[1, 0]],
            zs=[origin[2, 0], x_arr[2, 0]],
            color="red",
        )
        axis.plot(
            [origin[0, 0], y_arr[0, 0]],
            [origin[1, 0], y_arr[1, 0]],
            zs=[origin[2, 0], y_arr[2, 0]],
            color="green",
        )
        axis.plot(
            [origin[0, 0], z_arr[0, 0]],
            [origin[1, 0], z_arr[1, 0]],
            zs=[origin[2, 0], z_arr[2, 0]],
            color="blue",
        )
    else:
        if passed:
            axis.scatter(origin[0], origin[1], origin[2], color="green")
        else:
            axis.scatter(origin[0], origin[1], origin[2], color="red")
    axis.set_xbound(-0.4, 0.4)
    axis.set_ybound(-0.4, 0.4)
    axis.set_zlim(0, 0.6)


# def run_motion():

#     parse_pose()

if __name__ == "__main__":
    parse_pose("test.txt")
