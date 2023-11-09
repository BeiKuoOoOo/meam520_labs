import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))
    # Combine linear and angular velocities into a single end-effector velocity vector
    v_ee = np.concatenate((v_in, omega_in), axis=0)

    # Compute the Jacobian for the current configuration
    J = calcJacobian(q_in)
    # Concatenate linear and angular velocities into a single vector
    # and handle NaN values which represent unconstrained velocities
    desired_velocities = np.concatenate((v_in, omega_in))
    mask = ~np.isnan(desired_velocities)
    desired_velocities = desired_velocities[mask].reshape(-1, 1)
    J_masked = J[mask.ravel(), :]

    # Compute the pseudo-inverse of the masked Jacobian
    J_pinv = np.linalg.pinv(J_masked)

    # Compute the primary task joint velocities
    dq_primary = J_pinv @ desired_velocities

    # Compute the null space projector
    I = np.eye(J.shape[1])
    null_space_projector = I - J_pinv @ J_masked

    # Apply the secondary task in the null space of the Jacobian
    # Ensure that the secondary task does not interfere with the primary task
    dq_null = null_space_projector @ b

    # Combine primary and secondary tasks
    dq = dq_primary + dq_null

    # Flatten to 1 x 7 vector if necessary
    dq = dq.flatten()

    return dq

