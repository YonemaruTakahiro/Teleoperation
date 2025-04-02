import numpy as np

def abnormal_jnts_change_detection(current_jnt_values, next_jnt_values, threshold):
    jnts_change_degree = next_jnt_values - current_jnt_values
    abs_jnts_change_degree = np.abs(jnts_change_degree)
    for angle in abs_jnts_change_degree:
        if angle > threshold:
            return True
    return False