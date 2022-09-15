import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
from scipy.spatial import ConvexHull

np.random.seed(7)

class HiddenPointsRemoval():
	'''
		reference: 
			Katz, Sagi, Ayellet Tal, and Ronen Basri. "Direct Visibility of Point Sets." 2007. 
		github:
			https://github.com/williamsea/Hidden_Points_Removal_HPR/blob/master/MyHPR_HaiTang.py
	'''

	def __init__(self, points):
		self.points = points

	'''
	Function used to Perform Spherical Flip on the Original Point Cloud
	'''
	def sphericalFlip(self, center, param):

		n = len(self.points) # total n points
		points = self.points - np.repeat(center, n, axis = 0) # Move C to the origin
		normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
		# print(normPoints)
		# print(max(normPoints))
		R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere
		# print(R)
		# pdb.set_trace()
		R = np.repeat(1.5, n, axis = 0) # Radius of Sphere
		
		flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
		flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
		flippedPoints += points 

		return flippedPoints

	'''
	Function used to Obtain the Convex hull
	'''
	def convexHull(self, points):

		points = np.append(points, [[0,0,0]], axis = 0) # All points plus origin
		hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.

		return hull

	def plot(self, visible_hull_points):
		fig = plt.figure(figsize = plt.figaspect(0.5))
		plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')

		# First subplot
		ax = fig.add_subplot(1,2,1, projection = '3d')
		ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='r', marker='^') # Plot all points
		ax.set_xlabel('X Axis')
		ax.set_ylabel('Y Axis')
		ax.set_zlabel('Z Axis')

		# Second subplot
		ax = fig.add_subplot(1,2,2, projection = '3d')
		for vertex in visible_hull_points.vertices[:-1]:
			ax.scatter(self.points[vertex, 0], self.points[vertex, 1], self.points[vertex, 2], c='b', marker='o') # Plot visible points
		# ax.set_zlim3d(-1.5, 1.5)
		ax.set_xlabel('X Axis')
		ax.set_ylabel('Y Axis')
		ax.set_zlabel('Z Axis')

		plt.show()