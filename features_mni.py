"""Code to compute the feature matrix.
"""
import os
import numpy as np
import nibabel as nib
from os.path import isfile
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from distances import parallel_distance_computation
from functools import partial
from endpoints_distance import bundles_distances_endpoints_fastest
from waypoints_distance import wrapper_bundle2roi_distance, bundle2roi_distance
from subsampling import compute_subset
import pickle
import json
from dipy.tracking import metrics as tm
from dipy.tracking.metrics import endpoint
from sklearn.cluster import KMeans
from dipy.tracking.vox2track import streamline_mapping
from utils import compute_kdtree_and_dr_tractogram, compute_superset, NN, NN_radius

try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except ImportError:
    joblib_available = False


## features settings 
dm = True
extra_prototypes = False
local_prototypes = True
endpoints = True
rois = True
frenet_serret = False
fa_profile = False
context = False
context_hist = False

## global configuration parameters
distance_func = bundles_distances_mdf
num_local_prototypes = 100
nb_points = 20
with open('config.json') as f:
	data = json.load(f)
	tag = data["_inputs"][2]["datatype_tags"][0].encode("utf-8")


def compute_X_dm(superset, prototypes, distance_func=bundles_distances_mam, nb_points=20):
	"""Compute the global dissimilarity matrix.
	"""
	if distance_func==bundles_distances_mdf:
		print("Resampling the superset with %s points" %nb_points)
		superset = set_number_of_points(superset, nb_points)
	distance = partial(parallel_distance_computation, distance=distance_func)
	print("Computing dissimilarity matrix (%s x %s)..." %(len(superset), len(prototypes)))
	dm_superset = distance(superset, prototypes)
	
	return dm_superset


def compute_X_dm_local(superset, subjID, tract_name, distance_func=bundles_distances_mam, nb_points=20):
	"""Compute the local dissimilarity matrix.
	"""
	if distance_func==bundles_distances_mdf:
		print("Resampling the superset with %s points" %nb_points)
		superset = set_number_of_points(superset, nb_points)
	distance = partial(parallel_distance_computation, distance=distance_func)
	local_prot_fname = 'common_local_prototypes/%s_common_prototypes.npy' %tract_name
	local_prototypes = np.load(local_prot_fname)
	print("Computing dissimilarity matrix (%s x %s)..." %(len(superset), len(local_prototypes)))
	dm_local_superset = distance(superset, local_prototypes)
	
	return dm_local_superset


def compute_X_end(superset, prototypes):
	"""Compute the endpoint matrix.
	"""
	endpoint_matrix = bundles_distances_endpoints_fastest(superset, prototypes)
	endpoint_matrix = endpoint_matrix * 0.5
	return endpoint_matrix


def compute_X_roi(superset, subjID, tract_name, tag):
	"""Compute a matrix with dimension (len(superset), 2) that contains 
	   the distances of each streamline of the superset with the 2 ROIs. 
	""" 
	superset = set_number_of_points(superset, nb_points) #to speed up the computational time
	print("Loading the two-waypoint ROIs of the target...")
	table_filename = 'ROIs_labels_dictionary.pickle'
	table = pickle.load(open(table_filename))
	roi1_lab = table[tract_name].items()[0][1]
	roi2_lab = table[tract_name].items()[1][1]
	subjID = 'MNI'
	if (tag == 'afq'):
		roi_dir = 'templates_mni125'
		roi1_filename = '%s/sub-%s_var-AFQ_lab-%s_roi.nii.gz' %(roi_dir, subjID, roi1_lab)
		roi2_filename = '%s/sub-%s_var-AFQ_lab-%s_roi.nii.gz' %(roi_dir, subjID, roi2_lab)
	elif tag == 'wmaSeg':
		roi_dir = 'templates_mni125_ICBM2009c'
		roi1_filename = '%s/%s.nii.gz' %(roi_dir, roi1_lab)
		roi2_filename = '%s/%s.nii.gz' %(roi_dir, roi2_lab)
	roi1 = nib.load(roi1_filename)
	roi2 = nib.load(roi2_filename)
	print("Computing superset to ROIs distances...")
	if joblib_available:
		roi1_dist = wrapper_bundle2roi_distance(superset, roi1)
		roi2_dist = wrapper_bundle2roi_distance(superset, roi2)
	else:
		roi1_dist = bundle2roi_distance(superset, roi1)
		roi2_dist = bundle2roi_distance(superset, roi2)
	X_roi = np.vstack((roi1_dist, roi2_dist))

	return X_roi.T


def compute_endpoints(bundle):
	endpoints = np.zeros((len(bundle),3)) 
	for i, st in enumerate(bundle):
		endpoints[i] = endpoint(st)
	return endpoints


def orient_tract_kmeans(bundle):
	"""Ensure the startpoints to be always higher than the endpoints.
	"""
	points = compute_endpoints(bundle)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
	class0_up = (kmeans.cluster_centers_[0][2] > kmeans.cluster_centers_[1][2]) #constraint on the z axis
	oriented_bundle = []
	for i, st in enumerate(bundle):	
	    if kmeans.labels_[i]==class0_up:
			oriented_bundle.append(st)
	    else:
			tmp = np.flip(st, axis=0)
			oriented_bundle.append(tmp)
	return oriented_bundle


def compute_X_fs(superset, nb_points_fs=100):
	"""Compute a matrix with dimension (len(superset), 2*nb_points_fs) that 
	   contains the curvature and torsion profiles of the oriented superset.
	"""
	superset = set_number_of_points(superset, nb_points_fs)
	oriented_superset = orient_tract_kmeans(superset)
	#X_fs = np.zeros((len(superset), 2*nb_points_fs))
	#X_fs = np.zeros((len(superset), nb_points_fs)) #only curvature
	X_fs = np.zeros((len(superset), 2)) #mean
	for i, st in enumerate(oriented_superset):
		T,N,B,k,t = tm.frenet_serret(st)
		#X_fs[i,0:nb_points_fs] = k.T
		#X_fs[i,nb_points_fs:] = t.T
		X_fs[i,0] = np.mean(k)
		X_fs[i,1] = np.mean(t)
	return X_fs


def compute_X_fa(superset, exID, subjID, nb_points_fa=100):
	"""Compute a matrix with dimension (len(superset), nb_points_fa)  
	   that contains the FA profile of the oriented superset.
	"""
	superset = set_number_of_points(superset, nb_points_fa)
	oriented_superset = orient_tract_kmeans(superset)
	X_fa = np.zeros((len(superset), nb_points_fa))
	fa_nii = nib.load('%s/aligned_FA/FA_m%s_s%s.nii.gz' %(subjID, exID, subjID))
	aff = fa_nii.affine
	fa = fa_nii.get_data()
	for s in range(len(oriented_superset)-1):
		voxel_list = streamline_mapping(oriented_superset[s:s+1], affine=aff).keys()
		for v, (i,j,k) in enumerate(voxel_list):
			X_fa[s,v] = fa[i,j,k]
	return X_fa


def compute_X_context(X_tmp, superset, k=10):
	"""Add a second feature matrix composed of the features
	   of the k-Nearest-Neighbors.
	"""
	kdt, prototypes = compute_kdtree_and_dr_tractogram(superset, num_prototypes=40, 
									 distance_func=distance_func, nb_points=nb_points)
	X_context = np.zeros((len(X_tmp), k*len(X_tmp[0])))
	for i in range(len(superset)):
		nn_idx = compute_superset(superset[i:i+1], kdt, prototypes, k=k+1, distance_func=distance_func, nb_points=nb_points)
		nn_idx = nn_idx[1:] #remove the index corresdponding to itself
		nn_row = np.array([])
		for nn in nn_idx:
			nn_row = np.hstack([nn_row, X_tmp[nn,:]]) if nn_row.size else X_tmp[nn,:]
		X_context[i,:] = nn_row
	return X_context


def compute_X_context_hist(superset, k=100, r=40, n_bins=40):
	"""Add features of the k-Nearest-Neighbors.
	"""
	kdt, prototypes = compute_kdtree_and_dr_tractogram(superset, num_prototypes=40, 
									 distance_func=distance_func, nb_points=nb_points)
	X_context_hist = np.zeros((len(superset), n_bins))
	for i in range(len(superset)):
		streamline = superset[i:i+1]
		#D, I = NN(streamline, kdt, prototypes, k=k, distance_func=distance_func, nb_points=nb_points)
		I = NN_radius(streamline, kdt, prototypes, r=r, distance_func=distance_func, nb_points=nb_points)
		I = I[1:] #remove the index corresdponding to itself
		nn_str = superset[I]
		#Compute the exact distance
		if distance_func == bundles_distances_mdf:
			streamline = set_number_of_points(streamline, nb_points=nb_points)
			nn_str = set_number_of_points(nn_str, nb_points=nb_points)
		D = distance_func(streamline, nn_str)
		hist, bin_edges = np.histogram(D, bins=0.5*np.arange(n_bins+1), range=(0,0.5*n_bins))
		X_context_hist[i,:] = hist
		if i==0:
			print(hist)
	return X_context_hist


def compute_feature_matrix(superset, exID, subjID, tract_name, distance_func=distance_func, nb_points=nb_points):
	"""Compute the feature matrix.
	"""
	np.random.seed(0)
	feature_list = []

	if dm:
		if extra_prototypes:
			common_prototypes = np.load('sub-983773_var-ACPC_prototypes108.npy')
		else:
			common_prototypes = np.load('common_prototypes.npy')
		X_dm = compute_X_dm(superset, common_prototypes, distance_func=distance_func, nb_points=nb_points)
		feature_list.append(X_dm)
		print("----> Added dissimilarity matrix of size (%s, %s)" %(X_dm.shape))
		
	if local_prototypes:
		X_dm_local = compute_X_dm_local(superset, subjID, tract_name, distance_func=distance_func, nb_points=nb_points)
		feature_list.append(X_dm_local)
		print("----> Added local dissimilarity matrix of size (%s, %s)" %(X_dm_local.shape))

	if endpoints:
		if extra_prototypes:
			common_prototypes = np.load('sub-983773_var-ACPC_prototypes108.npy')
		else:
			common_prototypes = np.load('common_prototypes.npy')
		X_end = compute_X_end(superset, common_prototypes)
		feature_list.append(X_end)
		print("----> Added endpoint matrix of size (%s, %s)" %(X_end.shape))

	if rois:
		X_roi = compute_X_roi(superset, subjID, tract_name, tag)
		feature_list.append(X_roi)
		print("----> Added ROI distance matrix of size (%s, %s)" %(X_roi.shape))

	if frenet_serret:
		X_fs = compute_X_fs(superset)
		feature_list.append(X_fs)
		print("----> Added Frenet-Serret matrix of size (%s, %s)" %(X_fs.shape))

	if fa_profile:
		X_fa = compute_X_fa(superset, exID, subjID)	
		feature_list.append(X_fa)
		print("----> Added FA profile matrix of size (%s, %s)" %(X_fa.shape))

	#concatenation
	X_tmp = np.array([])
	for matrix in feature_list:
		X_tmp = np.hstack([X_tmp, matrix]) if X_tmp.size else matrix
	print("----> Size of final feature matrix: (%s, %s)" %(X_tmp.shape))

	if context:
		X_context = compute_X_context(X_tmp, superset)
		X_tmp = np.hstack([X_tmp, X_context])
		print("----> Size of final feature matrix with context: (%s, %s)" %(X_tmp.shape))
	elif context_hist:
		X_context_hist = compute_X_context_hist(superset)
		X_tmp = np.hstack([X_tmp, X_context_hist])
		print("----> Size of final feature matrix with context_hist: (%s, %s)" %(X_tmp.shape))

	return np.array(X_tmp, dtype=np.float32)
