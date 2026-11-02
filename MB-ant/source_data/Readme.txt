Ant Mushroom Body model for ant route following experiment
02/05/2015 by Fei Peng 
Any questions, please contact fei.peng@qmul.ac.uk
________________________________________________________________

1. Content description

—-antview—-folder contains the simulated environment and the original ant route data; 

—-result—-folder will be used for storing all the route following model test data. 

—-connection_generator.m—-generates random connections among Projection neurons and Kenyon Cells (with a rule of each KC receives on average 10 random PNs), and all-to-all connection among Kenyon Cells and the one output neuron (Extrinsic Neuron, EN).

—-ImgGrabber.m—-is a function to grab snapshot image from the simulated environment, with given locations in the world and heading angle. 

—-PN_neuron.m  KC_neuron.m  EN_neuron.m—-are Izhikevich type neuron functions to hold the specific Izhikevich parameters, specified in the manuscript. 

—-synapse.m—-is a generic function to do conductance-based synaptic transmission. 

—-PN_KC_synapse.m—-is the non-plastic conductance-based synapse between PNs and KCs. It holds specific parameters to call the ‘synapse.m’ function.

—-KC_EN_synapse.m—-is plastic conductance-based synapse, using Eligibility trace to tag pre-synaptic events, STDP and extracelluar biogenic amine level to modify the synaptic weights. 

—-GPU_network_fun.m—-is the main network function, given an input to PNs, this function will generate a corresponding EN output, and modify synaptic weight matrix if reward is provided.

—-STDP.m—-is the learning rule, at the moment it’s in an anti-hebbian form (decrease the weights no matter delta t < or > 0).

—-ant_route_following_test.m—-is the main script for running ant route following experiment, which contains three parts: 1) on-route image generation; 2) network training; 3) agent-based test in the simulated world.

————————————————————————————————————————————————————————————————

2. Hardware-requirement: this code uses MATLAB-CUDA GPU functions to speed up the network processing, so it ask for 

	1) relatively new versions of Matlab (e.g., 2013a or above), and Parallel computing toolbox - GPU computing (please check, for exmaple the availability of Matlab default function: ‘gpuDevice’).

	2) CUDA-enabled NVIDIA GPUs with compute capability 2.0 or higher. Please check the MATLAB GPU Computing webpage for further details:http://uk.mathworks.com/discovery/matlab-gpu.html

_______________________________________________

Please also note that it is possible to run the code on CPU-only devices, but that requires some modification of the code, e.g., change the gpuArray functions back to normal functions, and change the use of arrayfun or others into a proper vectorised version, in order to achieve high efficiency. 

