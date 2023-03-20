import numpy as np
import os
import open3d as o3d
import copy
# from draw_arrows import *


# https://iquilezles.org/articles/noacos/
def rotation_align(from_vec, to_vec):
    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)
    if np.all(v == np.zeros(3)) and c > 0:
        return np.eye(3)
    if np.all(v == np.zeros(3)) and c < 0:
        return -np.eye(3)
    k = 1.0 / (1.0 + c)

    return np.array([[v[0]**2 * k + c,    v[0]*v[1]*k - v[2], v[0]*v[2]*k + v[1]],
                     [v[0]*v[1]*k + v[2], v[1]**2 * k + c,    v[1]*v[2]*k - v[0]],
                     [v[0]*v[2]*k - v[1], v[1]*v[2]*k + v[0], v[2]**2 * k + c   ]])


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


# vec1 = np.random.rand(3)
# vec2 = np.random.rand(3)
# vec1 = vec1/np.linalg.norm(vec1)
# vec2 = vec2/np.linalg.norm(vec2)
# np.dot(vec1, vec2)
# R = rotation_align(vec1, vec2)
# vec3 = np.dot(R, vec2)
# np.dot(vec1, vec3)


#
# # calculate object center in WC
# # translate view to object center
# # translate WC to OC
# # rotate view
# # translate view back
#
# # do it first in open3d to get the matrices right
# knot_mesh = o3d.data.KnotMesh()
# mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
# mesh = mesh.filter_smooth_laplacian(number_of_iterations=50)
# mesh.compute_vertex_normals()
#
# box = mesh.get_axis_aligned_bounding_box()
# box.color = (1, 0, 0)
# # o3d.visualization.draw_geometries([box, mesh])
#
# # https://de.mathworks.com/help/vision/ug/camera-calibration.html
# # https://vita-group.github.io/Fall22/Lecture%2010+11+12.pdf
# # Maybe this
# # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
# # https://www.3dflow.net/elementsCV/S3.xhtml
#
# coord_w = o3d.geometry.TriangleMesh.create_coordinate_frame()
# coord_w.scale(20, center=coord_w.get_center())
#
# F = 500
# width = 640
# height = 480
# px = 1
# py = 1
# fx = F/px
# fy = F/py
# cx = width/2
# cy = height/2
# skew = 0
# intrinsic = np.array([[F/px, skew, cx],[0, F/py, cy], [0, 0, 1]])
# # cam = o3d.geometry.LineSet.create_camera_visualization(width, height, intrinsic, np.eye(4), scale=10.0)
# coord_cam = o3d.geometry.TriangleMesh.create_coordinate_frame()
# coord_cam.scale(10, center=coord_w.get_center())
#
# obj_center = mesh.get_center()
# coord_obj = o3d.geometry.TriangleMesh.create_coordinate_frame()
# coord_obj = copy.deepcopy(coord_obj).translate(mesh.get_center())
# coord_obj.scale(20, center=coord_obj.get_center())
#
# # o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam, cam])
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
# # Camera
# position = cam.get_center()
# direction = np.array([0, 0, 1])
# scale = 10
# cone_height = scale * 0.2
# cylinder_height = scale * 0.8
# cone_radius = scale / 50
# cylinder_radius = scale / 50
# arrow_F = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
#                                                     cone_height=cone_height,
#                                                     cylinder_radius=0.5,
#                                                     cylinder_height=cylinder_height)
# arrow_F.compute_vertex_normals()
# # o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam, cam, arrow_F])
#
#
# R_around_x = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
# R_around_Y = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
#
# arrow_fx = copy.deepcopy(arrow_F)
# arrow_fy = copy.deepcopy(arrow_F)
# arrow_fx.rotate(R_around_Y, center=np.zeros(3))
# arrow_fy.rotate(R_around_x, center=np.zeros(3))
# arrow_fx.scale(0.640, center=np.zeros(3))
# arrow_fy.scale(0.480, center=np.zeros(3))
# arrow_fy.translate((0, 0, 1*scale))
# arrow_fx.translate((0, 0, 1*scale))
#
# # o3d.visualization.draw_geometries([box, mesh, coord_obj, coord_cam, cam, arrow_F, arrow_fy, arrow_fx])
#
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
