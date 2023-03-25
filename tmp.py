import copy

import numpy as np

# Load data from file
data = np.load('/home/adrian/Documents/Nerf/256_to_512_fasttv/chair/ckpt.npz', allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

density_matrix = np.empty([512, 512, 512])
sh_matrix = np.empty([512, 512, 512, 9])

density_matrix = npy_density_data[npy_links.clip(min=0)]
sh_matrix = npy_sh_data[:,:9][npy_links.clip(min=0)]



# from tqdm import tqdm
# count = 0
# for i in tqdm(range(npy_links.shape[0])):
#     for j in range(npy_links.shape[1]):
#         for k in range(npy_links.shape[2]):
#             idx = npy_links[i, j, k]
#             if idx >= 0:
#                 # density_matrix[i, j, k] = npy_density_data[idx]
#                 sh_matrix[i, j, k] = npy_sh_data[idx, :9]
#                 count += 1

# density_matrix2 = npy_density_data[npy_links.clip(min=0)]
# density_matrix2 = np.squeeze(density_matrix2)
# mat = density_matrix2 == density_matrix
#
# npy_sh_data_red = npy_sh_data[:,:9]
# sh_matrix2 = npy_sh_data_red[npy_links.clip(min=0)]
# mat = sh_matrix2 == sh_matrix
#
# grid_density = np.zeros(npy_links.shape)
#
# tmp = npy_links >= 0
# grid_density[npy_links >= 0] = np.squeeze(npy_density_data[npy_links[npy_links >= 0]])
#
# path = '/home/adrian/Documents/Nerf/BlenderStuff/density'
# np.save(path, grid_density)

#https://www.youtube.com/watch?v=cqLhhjxch2s