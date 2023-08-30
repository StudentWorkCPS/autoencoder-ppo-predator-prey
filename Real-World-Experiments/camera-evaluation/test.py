import numpy as np
import matplotlib.pyplot as plt

def transform_to_real_space(pxl_points, real_points):
  """Transforms pixel points to real points.

  Args:
    pxl_points: A list of pixel points.
    real_points: A list of real points.

  Returns:
    A list of real points.
  """

  transformation_matrix = np.linalg.lstsq(
      pxl_points, real_points, rcond=-1)[0]
  
  np.save('projection.npy', transformation_matrix)
  real_points_transformed = np.dot(pxl_points, transformation_matrix)
  return real_points_transformed

if __name__ == "__main__":
    pxl_points = [[ 0.06905329, 0.51370734],
 [-0.53793436, -0.15157303],
 [ 0.6440696, -0.17446755],
 [ 0.03971611, -0.60741836],
 [-0.58147466,  0.44461304],
 [ 0.72620803,  0.41214412],
 [ 0.5137718,  -0.5856885 ],
 [-0.42026603, -0.56395143],
 [ 0.029024,   -0.1907939 ]]
    real_points = [[0, -188.9, 0], [-188.9, 0, 0], [188.1, 0, 0], [0, 188.4, 0], [-187.6, -189, 0], [188.8, -188.5, 0], [189, 187.8, 0], [-188, 188, 0], [0, 0, 0]]
    real_points_transformed = transform_to_real_space(pxl_points, real_points)
    print(real_points_transformed)

    ax = plt.axes(projection='3d')
    ax.scatter3D([x[0] for x in real_points_transformed], [x[1] for x in real_points_transformed], [x[2] for x in real_points_transformed])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter3D([x[0] for x in real_points], [x[1] for x in real_points], [x[2] for x in real_points])
    #ax.scatter3D([x[0] for x in pxl_points], [x[1] for x in pxl_points], [0 for x in pxl_points])
  
plt.show()
  