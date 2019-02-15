"""
pipeline_example.py
------------
Example pipeline for netrd
author: Tim LaRock
email: timothylarock at gmail dot com
Submitted as part of the 2019 NetSI Collabathon
"""

# NOTE: !IMPORTANT! If you want to play and make changes, 
# please make your own copy of this file (with a different name!) 
# first and edit that!!! Leave this file alone except to fix a bug!

import netrd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


## Load datasets
datasets = {'4-clique':netrd.utilities.read_time_series('../data/synth_4clique_N64_simple.csv'),
            'BA':netrd.utilities.read_time_series('../data/synth_BAnetwork_N64_simple.csv'),
            'ER':netrd.utilities.read_time_series('../data/synth_ERnetwork_N64_simple.csv')}


## Load reconstruction methods
reconstructors = {
                  'correlation_matrix':netrd.reconstruction.CorrelationMatrixReconstructor(),
                  'convergent_crossmappings':netrd.reconstruction.ConvergentCrossMappingReconstructor(),
                  'exact_mean_field':netrd.reconstruction.ExactMeanFieldReconstructor(),
                  'free_energy_minimization':netrd.reconstruction.FreeEnergyMinimizationReconstructor(),
                  'graphical_lasso':netrd.reconstruction.GraphicalLassoReconstructor(),
                  'maximum_likelihood':netrd.reconstruction.MaximumLikelihoodEstimationReconstructor(),
                  'mutual_information':netrd.reconstruction.MutualInformationMatrixReconstructor(),
                  'ou_inference':netrd.reconstruction.OUInferenceReconstructor(),
                  'partial_correlation':netrd.reconstruction.PartialCorrelationMatrixReconstructor(),
                  'regularized_correlation':netrd.reconstruction.RegularizedCorrelationMatrixReconstructor(),
                  'thouless_anderson_palmer':netrd.reconstruction.ThoulessAndersonPalmerReconstructor(),
                  'time_granger_causality':netrd.reconstruction.TimeGrangerCausalityReconstructor(),
                  'marchenko_pastur':netrd.reconstruction.MarchenkoPastur(),
                  #'naive_transfer_entropy':netrd.reconstruction.NaiveTransferEntropyReconstructor()
                 }



## Load distance methods
distance_methods = {'jaccard':netrd.distance.JaccardDistance(),
                    'hamming':netrd.distance.Hamming(),
                    'hamming_ipsen_mikhailov':netrd.distance.HammingIpsenMikhailov(),
                    #'portrait_divergence':netrd.distance.PortraitDivergence(),
                    #'resistance_perturbation':netrd.distance.ResistancePerturbation(),
                    'frobenius':netrd.distance.Frobenius(),
                    #'netsimilie':netrd.distance.NetSimile()
                   }


## get the names of the methods
reconstruction_methods = [method for method in reconstructors.keys()]
distance_methods_list = [method for method in distance_methods.keys()]

## Dictionary of dictionaries containing the reconstructed networks
## <dataset_name, <recon_method_name, reconstructed_graph>
networks = defaultdict(dict)

print('Computing network reconstructions')
## First get all of the reconstructions for every dataset
for data_name, time_series in datasets.items():
    print('dataset: ' + str(data_name))
    for reconstruction_method, reconstructor in reconstructors.items():
        print(reconstruction_method + '...', end='')
        networks[data_name][reconstruction_method] = reconstructor.fit(time_series)
        print('done')


## 4-deep dict structure: <dataset_name, <rmethod1, <rmethod2, <dmethod, distance> > > >  
distances = dict()
## In order to standardize, I am going to collect all of the 
## outputs for each distance
per_distance_values = dict()

print('Computing network distances')
## Then, compute the distance between every reconstruction of every network
for data_name, networks_dict in networks.items():
    per_distance_values[data_name] = defaultdict(list)
    print('dataset: ' + str(data_name))
    distances[data_name] = dict()
    for distance_method, distance_function in distance_methods.items():
        print(distance_method + '...', end='')
        for reconstruction_method1, network1 in networks_dict.items():
            distances[data_name].setdefault(reconstruction_method1, dict())
            for reconstruction_method2, network2 in networks_dict.items():
                distances[data_name][reconstruction_method1].setdefault(reconstruction_method2, dict())
                distance = distance_function.dist(network1, network2)
                distances[data_name][reconstruction_method1][reconstruction_method2].setdefault(distance_method, dict) 
                distances[data_name][reconstruction_method1][reconstruction_method2][distance_method] = distance
                per_distance_values[data_name][distance_method].append(distance)
        print('done')


## For each dataset and distance, store (max,min) tuple to use in standardization
max_min_distance_values = defaultdict(dict)
for data_name in networks.keys():
    for distance_method in distance_methods_list:
        max_min_distance_values[data_name][distance_method]=(np.max(per_distance_values[data_name][distance_method]), np.min(per_distance_values[data_name][distance_method]))


## Compute the similarity matrix by taking the average of the
## distance between every reconstruction matrix
number_of_reconstructors = len(reconstruction_methods)
name_map = {reconstruction_methods[i]:i for i in range(number_of_reconstructors)}
similarity_matrix = np.zeros((number_of_reconstructors,number_of_reconstructors))
for dataset, dataset_dict in distances.items():
    for method1, method1_dict in dataset_dict.items():
        for method2, method2_dict in dataset_dict.items():
            for distance_method in method1_dict[method2].keys():
                max_dist_val, min_dist_val = max_min_distance_values[data_name][distance_method]
                similarity_matrix[name_map[method1], name_map[method2]] += (method1_dict[method2][distance_method] - min_dist_val) / (max_dist_val - min_dist_val)

avg_similarity = similarity_matrix / (number_of_reconstructors*len(datasets))

print('Generating collabathon_output.png')
reconstruction_names = list(name_map.keys())
N_methods = len(reconstruction_names)
mat = avg_similarity
#### plotting parameters ####
netrd_cmap = 'bone_r'
method_id = 'test'
width = 1.2
heigh = 1.2
mult  = 8.0

###### plot the mat ###########
fig, ax0 = plt.subplots(1, 1, figsize=(width*mult,heigh*mult))

ax0.imshow(mat, aspect='auto', cmap=netrd_cmap)

###### be obsessive about it ###########
ax0.set_xticks(np.arange(0, N_methods, 1))
ax0.set_yticks(np.arange(0, N_methods, 1))
# ax0.set_xticklabels(np.arange(0, N_methods, 1), fontsize=2.0*mult)
# ax0.set_yticklabels(np.arange(0, N_methods, 1), fontsize=2.0*mult)
ax0.set_xticklabels(reconstruction_names, fontsize=1.5*mult, rotation=270)
ax0.set_yticklabels(reconstruction_names, fontsize=1.5*mult)
ax0.set_xticks(np.arange(-.5, N_methods-0.5, 1), minor=True)
ax0.set_yticks(np.arange(-.5, N_methods-0.5, 1), minor=True)
ax0.grid(which='minor', color='#333333', linestyle='-', linewidth=1.5)

ax0.set_title("Collabathon Fun Times Test Plot: \n Averaged Distance Between Reconstructed Networks", 
              fontsize=2.5*mult)

plt.savefig('collabathon_output.png', bbox_inches='tight', dpi=200)
