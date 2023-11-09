import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)
    ## STUDENT CODE STARTS HERE
    # Calculate the skew symmetric matrix of the rotation matrix difference
    S = 0.5 * (R_curr.T @ R_des - R_des.T @ R_curr)

    # Extract the coefficients of the skew symmetric matrix to form the rotation axis vector
    omega = np.array([S[2, 1], S[0, 2], S[1, 0]])

    # Calculate the rotation angle theta ensuring that the trace is within the valid range
    trace = np.trace(R_curr.T @ R_des)
    theta = np.arccos((trace - 1) / 2)

    # Normalize the rotation axis vector and scale by the sine of the rotation angle
    if not np.isclose(theta, 0):
        omega = omega / np.linalg.norm(omega) * np.sin(theta)
    else:
        omega = np.zeros(3)  # If theta is zero, then there is no rotation and omega is a zero vector.

    return omega