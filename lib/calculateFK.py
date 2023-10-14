import numpy as np
from math import pi


class FK():

    def __init__(self):
        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        joint1 = [0, 0, 0.141, 0]
        joint2 = [0, -pi / 2, 0.192, 0]
        joint3 = [0, pi / 2, 0, 0]
        joint4 = [0.0825, pi / 2, 0.195 + 0.121, 0]
        joint5 = [0.0825, pi / 2, 0, pi / 2 + pi / 2]
        joint6 = [0, -pi / 2, 0.125 + 0.259, 0]
        joint7 = [0.088, pi / 2, 0, -pi / 2 - pi / 2]
        joint8 = [0, 0, 0.051 + 0.159, -pi / 4]

        self.parameter = [joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8]

    def calculateA(self, a, alpha, d, theta):
        A = np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                      [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                      [0, np.sin(alpha), np.cos(alpha), d],
                      [0, 0, 0, 1]])
        return A

    def currentA(self, i, q):
        assert i >= 1
        currentParameter = self.parameter[i - 1]
        a = currentParameter[0]
        alpha = currentParameter[1]
        d = currentParameter[2]
        if i > 1:
            theta = currentParameter[3] + q[i - 2]
        else:
            theta = currentParameter[3]
        return self.calculateA(a, alpha, d, theta)

    def calculateDH(self, j, q):
        assert j >= 1
        for k in range(1, j + 1):
            if k == 1:
                H = self.currentA(k, q)
            else:
                H = H @ self.currentA(k, q)
        return H

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        jointPosition1 = (self.calculateDH(1, q) @ np.array([0, 0, 0, 1]))[:3]
        jointPosition2 = (self.calculateDH(2, q) @ np.array([0, 0, 0, 1]))[:3]
        jointPosition3 = (self.calculateDH(3, q) @ np.array([0, 0, 0.195, 1]))[:3]
        jointPosition4 = (self.calculateDH(4, q) @ np.array([0, 0, 0, 1]))[:3]
        jointPosition5 = (self.calculateDH(5, q) @ np.array([0, 0, 0.125, 1]))[:3]
        jointPosition6 = (self.calculateDH(6, q) @ np.array([0, 0, -0.015, 1]))[:3]
        jointPosition7 = (self.calculateDH(7, q) @ np.array([0, 0, 0.051, 1]))[:3]
        jointPosition8 = (self.calculateDH(8, q) @ np.array([0, 0, 0, 1]))[:3]

        jointPositions = np.stack(
            [jointPosition1, jointPosition2, jointPosition3, jointPosition4, jointPosition5, jointPosition6,
             jointPosition7, jointPosition8])
        assert jointPositions.shape == (8, 3)
        T0e = self.calculateDH(8, q)

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        axisList = []
        for i in range(1, 8):
            T = self.calculateDH(i, q)
            R = T[:3, :3]
            axis = R[:, 2]
            axis = axis / np.linalg.norm(axis)
            axisList.append(axis)
        axisarray = np.array(axisList).T

        return axisarray

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return ()


if __name__ == "__main__":
    fk = FK()

    # matches figure in the handout
    q = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    joint_positions, T0e = fk.forward(q)

    print("Joint Positions:\n", joint_positions)
    print("End Effector Pose:\n", T0e)
    print("rotation", fk.get_axis_of_rotation(q))
