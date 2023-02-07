from scipy.spatial import distance, distance_matrix
import numpy as np
from collections import Counter

src_corners = np.array([[0, 0],
                        [200, 0],
                        [0, 100],
                        [200, 100]])

corners = np.array([[175, 75],
                    [199, 75],
                    [175, 99],
                    [199, 99]], dtype=np.float32)


dst_mat = distance_matrix(src_corners, corners)
print(dst_mat)
print(np.argmin(dst_mat, axis=0))
print(np.argmin(dst_mat, axis=1))