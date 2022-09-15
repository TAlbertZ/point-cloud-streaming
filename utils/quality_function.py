import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt

RATE_UTILITY_EQ_6 = True
MODIFY_WITH_LOG = False
PROJECTION = False

tile_width = 1.8 / 16 # meter

rates = [127, 467, 1532]
distance = np.array([_/1 for _ in range(1,21)])
tile_a = 9.205676792183384
tile_b = 21.85296976213833

size_versions = np.array([12083,  3861,  1169,   408,   149,    58])
num_points_versions = np.array([5577, 1608,  423,  125,   36,    9])

# size_versions = np.array([3861,  1169,   408,   149,    58])
psnr_versions = np.array([35.80456667, 32.13946667, 28.74576667, 26.5047,     23.6522])


####################### x-axis: distance ##########################
# dist_c = 1
# legend = []
# for rate in rates:
# 	# quality = (tile_a * np.log(rate) + tile_b) / (1 + np.exp(-dist_c * distance))
# 	quality = tile_a * np.log(20 / distance) * np.log(rate) + tile_b * np.log(distance) * 5
# 	# first_derivative = -tile_a / 10 / np.power(distance, 2) * np.log(rate) + tile_b * 0.5 / np.power(distance, 0.5)
# 	# second_derivative = tile_a / 10 * 2 / np.power(distance, 3) * np.log(rate) - tile_b * 0.5 * 0.5 / np.power(distance, 1.5)
# 	# quality = (tile_a * np.log(rate) + tile_b) * np.power(distance, 0.5)
	
# 	# quality = (tile_a * np.log(size_versions) + tile_b) * num_points_versions

# 	plt.plot(distance, quality, linewidth=2)
# 	legend.append('size = %d Byte' %(rate))
# plt.legend(legend, fontsize=15, loc='best', ncol=1)
# plt.show()


####################### x-axis: distance ##########################
dist_c = 1
legend = []
for rate_idx in range(len(size_versions)):
	rate = size_versions[rate_idx]
	num_points = num_points_versions[rate_idx]
	# psnr = psnr_versions[rate_idx]
	# quality = (tile_a * np.log(rate) + tile_b) / (1 + np.exp(-dist_c * distance))
	quality = tile_a * np.log(20 / distance) * np.log(rate) + tile_b * np.log(distance) * 5
	# first_derivative = -tile_a / 10 / np.power(distance, 2) * np.log(rate) + tile_b * 0.5 / np.power(distance, 0.5)
	# second_derivative = tile_a / 10 * 2 / np.power(distance, 3) * np.log(rate) - tile_b * 0.5 * 0.5 / np.power(distance, 1.5)
	# quality = (tile_a * np.log(rate) + tile_b) * np.power(distance, 0.5)
	
	# quality = (tile_a * np.log(size_versions) + tile_b) * num_points_versions

	theta = tile_width / distance * 180 / np.pi # degree, shape: same as distance
	if RATE_UTILITY_EQ_6:
		if MODIFY_WITH_LOG:
			differentiable_num_points = num_points / np.power(theta, 2) # shape: (6, )
			quality = (tile_a * np.log(rate) + tile_b) * np.log(differentiable_num_points) # * np.power(theta, 2)
			quality = np.log(differentiable_num_points)
		else:
			num_points_per_degree = np.power(num_points, 0.5) / theta # shape: (6, )
			print(num_points_per_degree)
			num_points_per_degree[np.where(num_points_per_degree >= 60)] = 60
			differentiable_num_points = np.power(num_points_per_degree * theta, 2)
			quality = np.log(num_points_per_degree)
			quality = differentiable_num_points * np.log(rate)
			# quality = 4.6 * np.log(num_points_per_degree) + 2.9 + 3.6 * np.log10(tile_width / distance) + 2.7 * np.power(np.log10(num_points_per_degree), 2) - 1.7 * np.power(np.log10(num_points_per_degree), 3)
			print(quality)

	if PROJECTION:
		quality = (psnr * np.log10(np.power(theta, 2)) - 100 * np.log10(0.2 / distance))

	plt.plot(distance, quality, linewidth=2)
	legend.append('size = %d Byte' %(rate))
plt.legend(legend, fontsize=12, loc='best', ncol=1)
plt.xlabel('distance / m', fontsize=15, fontweight='bold')
plt.ylabel('utility', fontsize=15, fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.annotate("(5, 14.25)", (5, 14.2549644)) 
plt.plot([6], [14.49597285], marker="o", markersize=6, markerfacecolor="blue", markeredgecolor="None")
plt.text(6, 14.25, '(5, 14.50)', horizontalalignment='center', verticalalignment='top', fontweight='bold')
plt.plot([10], [13.69731735], marker="o", markersize=6, markerfacecolor="orange", markeredgecolor="None")
plt.text(10, 13.41226486, '(10, 13.70)', horizontalalignment='center', verticalalignment='top', fontweight='bold')
plt.plot([19], [12.69380438], marker="o", markersize=6, markerfacecolor="green", markeredgecolor="None")
plt.text(19, 12.45, '(19, 12.69)', horizontalalignment='center', verticalalignment='top', fontweight='bold')
plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
plt.tight_layout()
plt.show()




####################### x-axis: rate ##########################
# rates = np.array(range(500, 10000, 500))
distances = range(1, 11, 3)
# tile_a = 9.205676792183384
# tile_b = 21.85296976213833

dist_c = 1
legend = []
for distance in distances:
	# quality = (tile_a * np.log(rate) + tile_b) / (1 + np.exp(-dist_c * distance))
	quality = tile_a * np.log(20 / distance) * np.log(rates) + tile_b * np.log(distance) * 5
	# first_derivative = -tile_a / 10 / np.power(distance, 2) * np.log(rate) + tile_b * 0.5 / np.power(distance, 0.5)
	# second_derivative = tile_a / 10 * 2 / np.power(distance, 3) * np.log(rate) - tile_b * 0.5 * 0.5 / np.power(distance, 1.5)
	# quality = (tile_a * np.log(rate) + tile_b) * np.power(distance, 0.5)

	quality = (tile_a * np.log(size_versions) + tile_b) * num_points_versions

	if RATE_UTILITY_EQ_6:
		if MODIFY_WITH_LOG:
			theta = tile_width / distance * 180 / np.pi # degree
			differentiable_num_points_versions = num_points_versions / np.power(theta, 2) # shape: (6, )
			quality = (tile_a * np.log(size_versions) + tile_b) * np.log(differentiable_num_points_versions) * np.power(theta, 2)
		else:
			theta = tile_width / distance * 180 / np.pi # degree
			num_points_per_degree = np.power(num_points_versions, 0.5) / theta # shape: (6, )
			num_points_per_degree[np.where(num_points_per_degree >= 60)] = 60
			differentiable_num_points_versions = np.power(num_points_per_degree * theta, 2)
			quality = (tile_a * np.log(size_versions) + tile_b) * differentiable_num_points_versions #np.log(num_points_per_degree)
			# quality = 4.6 * np.log(num_points_per_degree) + 2.9 + 3.6 * np.log10(tile_width / distance) + 2.7 * np.power(np.log10(num_points_per_degree), 2) - 1.7 * np.power(np.log10(num_points_per_degree), 3)
			# quality = np.log(num_points_per_degree)

	if PROJECTION:
		pass

	# plt.plot(rates, quality, linewidth=2)
	plt.plot(size_versions, quality, linewidth=2)
	legend.append('distance = %dm' %(distance))
plt.legend(legend, fontsize=15, loc='best', ncol=1)
plt.xlabel('rate / Byte')
plt.ylabel('utility')
plt.show()