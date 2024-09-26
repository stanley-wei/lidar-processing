import numpy as np

def apply_filters(point_cloud, filters):
	for filter in filters:
		point_cloud = filter.filter(point_cloud)

	return point_cloud


def color_by_classification(point_cloud, classifications):
	cloud = pv.PolyData(point_cloud)
	cloud['point_color'] = classifications
	pv.plot(cloud, scalars='point_color')


def remap_classes(classifications, class_dict):
	remapped = np.zeros(classifications.shape, dtype=classifications.dtype)
	for key in class_dict.keys():
		remapped[np.nonzero(classifications == key)] = class_dict[key]

	return remapped
