import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    fk = FK()
    axis_of_rotation = fk.get_axis_of_rotation(q_in)
    joint_position,_ = fk.forward(q_in)

    # Calculate the Jacobian matrix
    for i in range(7):
        # Linear velocity component: cross product of joint axis and position vector
        J[:3, i] = np.cross(axis_of_rotation[:, i], np.array(joint_position[-1, :] - joint_position[i, :]))

        # Angular velocity component: joint axis
        J[3:, i] = axis_of_rotation[:, i]

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
