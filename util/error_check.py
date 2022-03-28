import numpy as np


def plane_check(plane_model):
    a, b, c, d = plane_model
    norm_vector = np.array([a, b, c])
    basic_vector = np.array(self.basic_plane[:3])
    norm_dist = np.sqrt(np.sum(np.power(norm_vector, 2)))
    basic_dist = np.sqrt(np.sum(np.power(basic_vector, 2)))

    norm_vector = norm_vector / norm_dist
    basic_vector = basic_vector / basic_dist

    norm_dist = np.sqrt(np.sum(np.power(norm_vector, 2)))
    basic_dist = np.sqrt(np.sum(np.power(basic_vector, 2)))
    print("dist")
    print(norm_dist, basic_dist)

    check_theta = np.dot(norm_vector, basic_vector)
    check_theta = np.arccos(check_theta)

    print('check_theta', check_theta)
    print('check_theta', check_theta * (180 / np.pi))

    assert -0.2 < check_theta < 0.2, f"{check_theta}"
