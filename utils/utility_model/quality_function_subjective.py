import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
import matplotlib.pyplot as plt

TILE_WIDTH = 1.8 / 16  # meter
MAX_NUM_POINTS_IN_TILE = 5577
NUM_DOWNSAMPLE_TYPES = 4  # 0.2, 0.4, 0.6, 0.8
DOWNSAMPLE_PERCENTS = [20, 40, 60, 80]
NUM_SAMPLE_PAIRS = 100 * 5  # for every downsample level
NUM_SAMPLES = NUM_SAMPLE_PAIRS * NUM_DOWNSAMPLE_TYPES

files = []
ssim = []
distances = []
num_points = []

for downsample_level in DOWNSAMPLE_PERCENTS:
    files.append(
        open("../SSIM-Distance/SSIM-Dis_" + str(downsample_level) + ".txt",
             "r"))

for sample_idx in range(NUM_SAMPLE_PAIRS):
    for downsample_level_idx in range(len(DOWNSAMPLE_PERCENTS)):
        line = files[downsample_level_idx].readline()
        # print(line.split())
        # pdb.set_trace()
        line_list = line.split()
        ssim.append(float(line_list[0]))
        distances.append(float(line_list[1]))
        num_points.append(MAX_NUM_POINTS_IN_TILE *
                          DOWNSAMPLE_PERCENTS[downsample_level_idx] /
                          100)  # downsample_level is percentage

# pdb.set_trace()

ssim = np.array(ssim)
distances = np.array(distances)
num_points = np.array(num_points)

thetas_radian = TILE_WIDTH / distances  # radian
thetas_degree = thetas_radian * 180 / np.pi  # degree
log_thetas = np.log10(thetas_radian)
angular_freqs = np.power(num_points, 0.5) / thetas_degree  # per degree
angular_freqs[np.where(angular_freqs >= 60)] = 60
log_angular_freqs = np.log10(angular_freqs)
log_angular_freqs_square = np.power(log_angular_freqs, 2)
log_angular_freqs_cubic = np.power(log_angular_freqs, 3)

########### fit ################
X = []
for sample_idx in range(NUM_SAMPLES):
    X.append([])
    X[sample_idx].append(log_thetas[sample_idx])
    X[sample_idx].append(log_angular_freqs[sample_idx])
    X[sample_idx].append(log_angular_freqs_square[sample_idx])
    X[sample_idx].append(log_angular_freqs_cubic[sample_idx])

reg = LinearRegression().fit(X, ssim)
coefs = reg.coef_
intercept = reg.intercept_
print(coefs)
print(intercept)
# pdb.set_trace()

distance = np.array([_ / 1 for _ in range(1, 6)])
distance = np.array([0.8, 1, 2, 3, 4, 5])
size_versions = np.array([12083, 3861, 1169, 408, 149, 58])
num_points_versions = np.array([5577, 1608, 423, 125, 36, 9])

size_versions = np.array([2336.,  869.,  306.,  119.,   50.,   31.])
num_points_versions = np.array([2592.,  762.,  214.,   61.,   21.,    8.])

size_versions = np.array([3858., 1462.,  508.,  187.,   86.,   39.])
num_points_versions = np.array([5626., 1674.,  456.,  123.,   44.,   13.])

size_versions = np.array([76., 46., 32., 23., 24., 21.])
num_points_versions = np.array([75., 32., 11.,  2.,  2.,  1.])

# size_versions = np.array([1718.,  716.,  269.,  105.,   52.,   34.])
# num_points_versions = np.array([1843.,  592.,  174.,   54.,   18.,    7.])


legend = []
for rate_idx in range(len(size_versions)):
    theta = TILE_WIDTH / distance * 180 / np.pi  # degree, shape: same as distance
    rate = size_versions[rate_idx]
    num_points = num_points_versions[rate_idx]
    num_points_per_degree = np.power(num_points, 0.5) / theta  # shape: (6, )
    print(num_points_per_degree)
    num_points_per_degree[np.where(num_points_per_degree >= 60)] = 60
    differentiable_num_points = np.power(num_points_per_degree * theta, 2)
    quality = np.log(num_points_per_degree)
    quality = differentiable_num_points * np.log(rate)
    quality = coefs[1] * np.log10(num_points_per_degree) + intercept + coefs[
        0] * np.log10(TILE_WIDTH / distance) + coefs[2] * np.power(
            np.log10(num_points_per_degree), 2) + coefs[3] * np.power(
                np.log10(num_points_per_degree), 3)
    print(quality)

    plt.plot(distance, quality, linewidth=2)
    legend.append('size = %d Byte' % (rate))
plt.legend(legend, fontsize=12, loc='best', ncol=1)
plt.xlabel('distance / m', fontsize=15, fontweight='bold')
plt.ylabel('utility', fontsize=15, fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
# plt.show()
plt.savefig("utility_distance_test.eps", bbox_inches='tight')

plt.close()
####################### x-axis: rate ##########################
# rates = np.array(range(500, 10000, 500))
distances = range(1, 6)
distances = [0.8, 1, 2, 3, 4, 5]
# tile_a = 9.205676792183384
# tile_b = 21.85296976213833

dist_c = 1
legend = []
for distance in distances:
    theta = TILE_WIDTH / distance * 180 / np.pi  # degree
    num_points_per_degree = np.power(num_points_versions,
                                     0.5) / theta  # shape: (6, )
    num_points_per_degree[np.where(num_points_per_degree >= 60)] = 60
    differentiable_num_points_versions = np.power(
        num_points_per_degree * theta, 2)
    # quality = 4.6 * np.log(num_points_per_degree) + 2.9 + 3.6 * np.log10(tile_width / distance) + 2.7 * np.power(np.log10(num_points_per_degree), 2) - 1.7 * np.power(np.log10(num_points_per_degree), 3)

    quality = coefs[1] * np.log10(num_points_per_degree) + intercept + coefs[
        0] * np.log10(TILE_WIDTH / distance) + coefs[2] * np.power(
            np.log10(num_points_per_degree), 2) + coefs[3] * np.power(
                np.log10(num_points_per_degree), 3)
    plt.plot(size_versions, quality, linewidth=2)
    legend.append('distance = ' + str(distance) + 'm')
plt.legend(legend, fontsize=15, loc='best', ncol=1)
plt.xlabel('rate / Byte')
plt.ylabel('utility')
# plt.show()
plt.savefig("utility_rate_test.eps", bbox_inches='tight')
