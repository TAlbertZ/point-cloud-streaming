import numpy as np
import pdb
import matplotlib.pyplot as plt

RATE_UTILITY_EQ_6 = False
MODIFY_WITH_LOG = False
PROJECTION = False

tile_width = 1.8 / 16 # meter

rates = [127, 467, 1532]
distance = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
# distance = np.array(range(1, 11))
tile_a = 9.205676792183384
tile_b = 21.85296976213833

size_versions = np.array([12083,  3861,  1169,   408,   149,    58])
num_points_versions = np.array([5577, 1608,  423,  125,   36,    9])

# size_versions = np.array([3861,  1169,   408,   149,    58])
psnr_versions = np.array([35.80456667, 32.13946667, 28.74576667, 26.5047,     23.6522])
# num_points_versions = np.array([1608,  423,  125,   36,    9])


###################### x-axis: distance ##########################
dist_c = 1
legend = []
for rate_idx in range(len(size_versions)):
	rate = size_versions[rate_idx]
	num_points = num_points_versions[rate_idx]
	theta_tile = tile_width / distance
	angular_resolution = np.power(num_points, 1) / theta_tile / 180 * np.pi

	# angular_resolution[np.where(angular_resolution > 60)] = 60

	picture_angle = tile_width / distance

	log_picture_angle = np.log(picture_angle)
	log_angular_resolution = np.log(angular_resolution)
	print(log_picture_angle)



	# quality = (tile_a * np.log(rate) + tile_b) / (1 + np.exp(-dist_c * distance))
	quality = tile_a * np.log(20 / distance) * np.log(rate) + tile_b * np.log(distance) * 5
	# first_derivative = -tile_a / 10 / np.power(distance, 2) * np.log(rate) + tile_b * 0.5 / np.power(distance, 0.5)
	# second_derivative = tile_a / 10 * 2 / np.power(distance, 3) * np.log(rate) - tile_b * 0.5 * 0.5 / np.power(distance, 1.5)
	# quality = (tile_a * np.log(rate) + tile_b) * np.power(distance, 0.5)
	
	# quality = (tile_a * np.log(size_versions) + tile_b) * num_points_versions
	quality = 3.6 * log_picture_angle + 2.9 + 4.6 * log_angular_resolution # + 2.7 * np.power(log_angular_resolution, 2) - 1.7 * np.power(log_angular_resolution, 3)

	plt.plot(distance, quality, linewidth=2)
	legend.append('size = %d Byte' %(rate))
plt.legend(legend, fontsize=15, loc='best', ncol=1)
plt.xlabel('distance / m')
plt.ylabel('utility')
plt.show()

