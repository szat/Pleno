import numpy as np
import os
import open3d as o3d
import copy


# https://iquilezles.org/articles/noacos/
def rotation_align(from_vec, to_vec):
    assert from_vec.shape == to_vec.shape, "from_vec and to_vec need to be of the same shape"
    if from_vec.ndim == 1:
        v = np.cross(from_vec, to_vec)
        # c = np.einsum('ij,ij...->i...', from_vec, to_vec)
        c = np.dot(from_vec, to_vec)
        if np.all(v == np.zeros(3)) and c > 0:
            return np.eye(3)
        if np.all(v == np.zeros(3)) and c < 0:
            return -np.eye(3)
        k = 1.0 / (1.0 + c)
        return np.array([[v[0]**2 * k + c,    v[0]*v[1]*k - v[2], v[0]*v[2]*k + v[1]],
                         [v[0]*v[1]*k + v[2], v[1]**2 * k + c,    v[1]*v[2]*k - v[0]],
                         [v[0]*v[2]*k - v[1], v[1]*v[2]*k + v[0], v[2]**2 * k + c   ]])
    if from_vec.ndim == 2:
        v = np.cross(from_vec, to_vec)
        c = np.einsum('ij,ij...->i...', from_vec, to_vec)
        k = 1.0 / (1.0 + c)
        out = np.array([[v[:, 0]**2 * k + c,    v[:, 0]*v[:, 1]*k - v[:, 2], v[:, 0]*v[:, 2]*k + v[:, 1]],
                         [v[:, 0]*v[:, 1]*k + v[:, 2], v[:, 1]**2 * k + c,    v[:, 1]*v[:, 2]*k - v[:, 0]],
                         [v[:, 0]*v[:, 2]*k - v[:, 1], v[:, 1]*v[:, 2]*k + v[:, 0], v[:, 2]**2 * k + c   ]])
        out = np.einsum('ijk->kij', out) # rearrange dimensions
        bool_flag_identity = np.all(v == np.zeros(3), axis=1) * c > 0
        bool_flag_reverse = np.all(v == np.zeros(3), axis=1) * c < 0
        out[bool_flag_identity] = np.eye(3)
        out[bool_flag_reverse] = -np.eye(3)
        return out


class Camera:
    def __init__(self,
                 origin=np.zeros(3),
                 orientation=np.array([0, 0, 1]),
                 dist_plane=1,
                 length_x=0.640,
                 length_y=0.480,
                 pixels_x=640,
                 pixels_y=480):
        self.origin = origin
        self.orientation = orientation
        self.dist_plane = dist_plane
        self.length_x = length_x
        self.length_y = length_y
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y


def get_arrow(start_point, end_point, color=[0.3, 0.3, 0.3]):
    vec = end_point - start_point
    norm = np.linalg.norm(vec)
    cone_height = norm * 0.2
    cylinder_height = norm * 0.8
    cone_radius = 0.2
    cylinder_radius = 0.1
    arrow = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
                                                     cone_height=cone_height,
                                                     cylinder_radius=cylinder_radius,
                                                     cylinder_height=cylinder_height)
    vec = vec / norm
    R = rotation_align(np.array([0, 0, 1]), vec)
    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(start_point)
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color)
    return arrow


def get_camera_vectors(camera: Camera):
    z = np.array([0, 0, 1])
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    # Put into world coordinates
    R = rotation_align(z, camera.orientation)
    z = np.dot(R, z)
    x = np.dot(R, x)
    y = np.dot(R, y)

    # Camera template
    z = z * camera.dist_plane
    x = x * camera.length_x
    y = y * camera.length_y

    z = z + camera.origin
    x = x + camera.origin
    y = y + camera.origin
    return z, x, y


# do it first in open3d to get the matrices right
knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
mesh = mesh.filter_smooth_laplacian(number_of_iterations=50)
mesh.compute_vertex_normals()
#
box = mesh.get_axis_aligned_bounding_box()
box.color = (1, 0, 0)
obj_center = mesh.get_center()
coord_obj = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_obj = copy.deepcopy(coord_obj).translate(mesh.get_center())
coord_obj.scale(20, center=coord_obj.get_center())
o3d.visualization.draw_geometries([box, mesh, coord_obj])

coord_w = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_w.scale(20, center=coord_w.get_center())
o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_w])

obj_center = obj_center / np.linalg.norm(obj_center)
camera = Camera(origin=np.zeros(3), orientation=obj_center, dist_plane=10, length_x=6.4, length_y=4.8)
z, x, y = get_camera_vectors(camera)
cam_z = get_arrow(np.zeros(3), z, [0, 0, 1]) #blue
cam_x = get_arrow(np.zeros(3)+z, x+z, [1, 0, 0]) #red
cam_y = get_arrow(np.zeros(3)+z, y+z, [0, 1, 0]) #green
o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_w, cam_z, cam_x, cam_y])

F = 500
width = 640
height = 480
px = 1
py = 1
fx = F/px
fy = F/py
cx = width/2
cy = height/2
skew = 0
intrinsic = np.array([[F/px, skew, cx],[0, F/py, cy], [0, 0, 1]])
cam = o3d.geometry.LineSet.create_camera_visualization(width, height, intrinsic, np.eye(4), scale=10.0)
o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_w, cam_z, cam_x, cam_y, cam])

vec = coord_obj.get_center() - cam.get_center()
vec = vec/np.linalg.norm(vec)
R = rotation_align(np.array([0, 0, 1]), vec)
cam.rotate(R, center=coord_w.get_center())
o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_w, cam_z, cam_x, cam_y, cam])










# # https://de.mathworks.com/help/vision/ug/camera-calibration.html
# # https://vita-group.github.io/Fall22/Lecture%2010+11+12.pdf
# # Maybe this
# # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
# # https://www.3dflow.net/elementsCV/S3.xhtml
#

F = 500
width = 640
height = 480
px = 1
py = 1
fx = F/px
fy = F/py
cx = width/2
cy = height/2
skew = 0
intrinsic = np.array([[F/px, skew, cx],[0, F/py, cy], [0, 0, 1]])
cam = o3d.geometry.LineSet.create_camera_visualization(width, height, intrinsic, np.eye(4), scale=10.0)
coord_cam = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_cam.scale(10, center=coord_w.get_center())
#
obj_center = mesh.get_center()
coord_obj = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_obj = copy.deepcopy(coord_obj).translate(mesh.get_center())
coord_obj.scale(20, center=coord_obj.get_center())

o3d.visualization.draw_geometries([box, mesh, coord_obj, cam, coord_cam])





#
# vec = coord_obj.get_center() - coord_cam.get_center()
# vec = vec/np.linalg.norm(vec)
# R = rotation_align(vec, np.array([0, 0, 1]))
#
# coord_cam2 = copy.deepcopy(coord_cam)
# coord_cam2.rotate(R, center=coord_cam.get_center())
#
# cam2 = copy.deepcopy(cam)
# cam2.rotate(R, center=coord_cam.get_center())
#
# # o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam2, cam2])
#
# Camera
position = cam.get_center()
direction = np.array([0, 0, 1])
scale = 10
cone_height = scale * 0.2
cylinder_height = scale * 0.8
cone_radius = scale / 50
cylinder_radius = scale / 50
arrow_F = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
                                                    cone_height=cone_height,
                                                    cylinder_radius=0.5,
                                                    cylinder_height=cylinder_height)
arrow_F.compute_vertex_normals()
o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam, cam, arrow_F])

R_around_x = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
R_around_y = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))

Rx = rotation_align(np.array([0, 0, 1]), np.array([0, 1, 0]))
np.testing.assert_allclose(Rx, R_around_x, atol=1e-10)

Ry = rotation_align(np.array([0, 0, 1]), np.array([1, 0, 0]))
np.testing.assert_allclose(Ry, R_around_y, atol=1e-10)


Rx * np.array([0, 0, 1])


arrow_fx = copy.deepcopy(arrow_F)
arrow_fy = copy.deepcopy(arrow_F)
arrow_fx.rotate(R_around_y, center=np.zeros(3))
arrow_fy.rotate(R_around_x, center=np.zeros(3))
arrow_fx.scale(0.640, center=np.zeros(3))
arrow_fy.scale(0.480, center=np.zeros(3))
arrow_fy.translate((0, 0, 1*scale))
arrow_fx.translate((0, 0, 1*scale))

o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam, cam, arrow_F, arrow_fy, arrow_fx])

arrow_x = get_arrow(np.array([0, 0, 0]), np.array([10, 0, 0]))
arrow_y = get_arrow(np.array([0, 0, 0]), np.array([0, 10, 0]))
arrow_z = get_arrow(np.array([0, 0, 0]), np.array([0, 0, 10]))
arrow_d = get_arrow(np.array([10, 10, 10]), np.array([20, 20, 20]))
o3d.visualization.draw_geometries([box, mesh, coord_w, arrow_x, arrow_y, arrow_z, arrow_d])

# arrow_fx_R = copy.deepcopy(arrow_fx)
# arrow_fy_R = copy.deepcopy(arrow_fy)
# arrow_F_R = copy.deepcopy(arrow_F)
# cam_R = copy.deepcopy(cam)
# coord_cam_R = copy.deepcopy(coord_cam)
#
#
# arrow_fx_R.rotate(R, center=np.zeros(3))
# arrow_fy_R.rotate(R, center=np.zeros(3))
# arrow_F_R.rotate(R, center=np.zeros(3))
# cam_R.rotate(R, center=np.zeros(3))
# coord_cam_R.rotate(R, center=np.zeros(3))
#
# # o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam, cam_R, arrow_F_R, arrow_fy_R, arrow_fx_R])
#
# # Position: the location of the camera in 3D space
# # Orientation: the direction that the camera is pointing in 3D space
# # Field of view: the angle of the cone of vision that the camera captures
# # Aspect ratio: the ratio of the width to the height of the image plane
# # Near and far planes: the distances from the camera at which objects are visible
#
