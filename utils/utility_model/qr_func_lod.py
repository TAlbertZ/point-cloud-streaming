import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
import matplotlib.pyplot as plt
import pickle as pk

TILE_WIDTH = 1.8 / 16  # meter
distances = []
num_points = []


# size_versions = np.array([12083, 3861, 1169, 408, 149, 58])
# num_points_versions = np.array([5577, 1608, 423, 125, 36, 9])
# 
# size_versions = np.array([2336.,  869.,  306.,  119.,   50.,   31.])
# num_points_versions = np.array([2592.,  762.,  214.,   61.,   21.,    8.])
# 
# size_versions = np.array([3858., 1462.,  508.,  187.,   86.,   39.])
# num_points_versions = np.array([5626., 1674.,  456.,  123.,   44.,   13.])
# 
# size_versions = np.array([76., 46., 32., 23., 24., 21.])
# num_points_versions = np.array([75., 32., 11.,  2.,  2.,  1.])

# size_versions = np.array([1718.,  716.,  269.,  105.,   52.,   34.])
# num_points_versions = np.array([1843.,  592.,  174.,   54.,   18.,    7.])

with open('../../../psnr_weights/results_map_tileRate_to_lod/new_a_300x16x16x16.pkl', 'rb') as file:
    tile_a = pk.load(file)
file.close()
with open('../../../psnr_weights/results_map_tileRate_to_lod/new_b_300x16x16x16.pkl', 'rb') as file:
    tile_b = pk.load(file)
file.close()

with open('../../../psnr_weights/results_map_tileRate_to_lod/rate_versions_6x300x16x16x16.pkl', 'rb') as file:
    size_versions_all_tiles = pk.load(file)
file.close()

nonzero_frame_idx, nonzero_x, nonzero_y, nonzero_z = np.sum(size_versions_all_tiles, axis=0).nonzero()

for tile_idx in range(10):
    distances = np.array([0.8, 1, 2, 3, 4, 5])
    frame_idx = nonzero_frame_idx[tile_idx]
    x = nonzero_x[tile_idx]
    y = nonzero_y[tile_idx]
    z = nonzero_z[tile_idx]
    size_versions = size_versions_all_tiles[:, frame_idx, x, y, z]

    a = tile_a[frame_idx][x][y][z]
    b = tile_b[frame_idx][x][y][z]

    legend = []
    for rate_idx in range(len(size_versions)):
        theta = TILE_WIDTH / distances * 180 / np.pi  # degree, shape: same as distances
        rate = size_versions[rate_idx]
        # num_points = num_points_versions[rate_idx]
        # num_points_per_degree = np.power(num_points, 0.5) / theta  # shape: (6, )

        lod = a * np.log(b*rate+1)
        num_points_per_degree = 2**lod / theta # shape: (6, )

        # with saturation cap effect
        num_points_per_degree[np.where(num_points_per_degree >= 60)] = 60

        print(num_points_per_degree)
        quality = np.log(num_points_per_degree)
        print(quality)

        plt.plot(distances, quality, linewidth=2)
        legend.append('size = %d Byte' % (rate))
    plt.legend(legend, fontsize=10, loc='best', ncol=1)
    plt.xlabel('distance / m', fontsize=15, fontweight='bold')
    plt.ylabel('utility', fontsize=15, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.show()
    plt.savefig("results_utility_func/cap_utility_distance" + str(tile_idx) + ".eps", bbox_inches='tight')

    plt.close()
    ####################### x-axis: rate ##########################
    # rates = np.array(range(500, 10000, 500))
    distances = [0.8, 1, 2, 3, 4, 5]

    legend = []
    for distance in distances:
        theta = TILE_WIDTH / distance * 180 / np.pi  # degree
        lod = a*np.log(b*size_versions+1)
        num_points_per_degree = 2**lod / theta # shape: (6, )

        # with saturation cap effect
        num_points_per_degree[np.where(num_points_per_degree >= 60)] = 60

        quality = np.log(num_points_per_degree)
        plt.plot(size_versions, quality, linewidth=2)
        legend.append('distance = ' + str(distance) + 'm')
    plt.legend(legend, fontsize=10, loc='best', ncol=1)
    plt.xlabel('rate / Byte')
    plt.ylabel('utility')
    # plt.show()
    plt.savefig("results_utility_func/cap_utility_rate" + str(tile_idx) + ".eps", bbox_inches='tight')
    plt.close()
