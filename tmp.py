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

grid_density = np.zeros(npy_links.shape)

tmp = npy_links >= 0
grid_density[npy_links >= 0] = np.squeeze(npy_density_data[npy_links[npy_links >= 0]])

path = '/home/adrian/Documents/Nerf/BlenderStuff/density'
np.save(path, grid_density)