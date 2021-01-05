from n3ml.n3ml.Network import Network
from n3ml.n3ml.Source import MNISTSource2, Source, MNISTSource
from n3ml.n3ml.Population import SRMPopulation, LIFPopulation, Population
from n3ml.n3ml.Connection import Connection, STDPConnection
from n3ml.n3ml.Learning import SpikeProp, STDP, Competitive_NormAD
from n3ml.n3ml.BuilderDir.Operators import *
import numpy as np


class Builder:
    @classmethod
    def build(cls, model, obj, sampling_period):
        if isinstance(obj, Network):
            return build_network(model, obj, sampling_period)

        raise NotImplementedError

def build_network(model, network, sampling_period):
    '''Build network'''

    '''Build running_period'''
    model.signal['threshold'] = np.array(network.threshold)
    model.signal['current_time'] = np.array(0)
    model.signal['current_period'] = np.array(0)

    model.add_op(UpdateTime(current_time=model.signal['current_time']))
    model.add_op(UpdatePeriod(current_period=model.signal['current_period'],
                              sampling_period=sampling_period))

    '''Build source and population'''
    for obj in network.source + network.population:
        if isinstance(obj, MNISTSource):
            build_mnistsource(model, obj, sampling_period)

        elif isinstance(obj, LIFPopulation):
            build_lif_population(model, obj)
        elif isinstance(obj, SRMPopulation):
            build_srm_population(model, obj)

    '''Build connection'''
    for obj in network.connection:
        if isinstance(obj, Connection):
            build_connection(model, obj, network)

    '''Build learning algorithm'''
    if isinstance(network.learning, STDP):
        pass
    elif isinstance(network.learning, SpikeProp):
        build_spikeprop(model, network, sampling_period)
    elif isinstance(network.learning, Competitive_NormAD):
        pass


def build_mnistsource(model, mnistsource, sampling_period):
    # Signals
    model.signal[mnistsource] = {}
    model.signal[mnistsource]['image'] = np.zeros(
        shape=(mnistsource.rows, mnistsource.cols))
    model.signal[mnistsource]['image_index'] = np.array(-1)
    model.signal[mnistsource]['label'] = np.array(0)
    model.signal[mnistsource]['label_index'] = np.array(-1)

    # Operators
    model.add_op(SampleLabel(label=model.signal[mnistsource]['label'],
                             label_index=model.signal[mnistsource]['label_index'],
                             current_period=model.signal['current_period'],
                             labels=mnistsource.labels))


    model.add_op(SampleImage(image=model.signal[mnistsource]['image'],
                              image_index=model.signal[mnistsource]['image_index'],
                              current_period=model.signal['current_period'],
                              images=mnistsource.images))

    if mnistsource.code == 'population':
        model.signal[mnistsource]['spike_time'] = np.zeros(
            shape=(mnistsource.rows * mnistsource.cols, mnistsource.num_neurons))

        model.add_op(InitSpikeTime(spike_time=model.signal[mnistsource]['spike_time'],
                                             current_period=model.signal['current_period'],
                                             value=model.nan))

        model.add_op(PopulationEncode(image=model.signal[mnistsource]['image'],
                                                spike_time=model.signal[mnistsource]['spike_time'],
                                                num_neurons=mnistsource.num_neurons,
                                                interval=mnistsource.sampling_period,
                                                beta=mnistsource.beta,
                                                min_value=mnistsource.min_value,
                                                max_value=mnistsource.max_value,
                                                sampling_period=sampling_period))


def build_srm_population(model, srmpopulation):
    model.signal[srmpopulation] = {}

    model.signal[srmpopulation]['membrane_potential'] = np.zeros(shape=srmpopulation.num_neurons)
    model.signal[srmpopulation]['spike_time'] = np.zeros(shape=srmpopulation.num_neurons)
    model.signal[srmpopulation]['tau'] = np.array(srmpopulation.tauLeak)



    model.add_op(InitSpikeTime(spike_time=model.signal[srmpopulation]['spike_time'],
                               current_period=model.signal['current_period'],
                               value=model.nan))

    model.add_op(SpikeTime(membrane_potential=model.signal[srmpopulation]['membrane_potential'],
                           spike_time=model.signal[srmpopulation]['spike_time'],
                           threshold=model.signal['threshold'],
                           current_period=model.signal['current_period']))



def build_connection(model, connection, network):

    if isinstance(connection.pre, Population):
        pre_num_neurons = connection.pre.num_neurons
        print(pre_num_neurons)
    elif isinstance(connection.pre, MNISTSource):
        pre_num_neurons = connection.pre.rows * connection.pre.cols * connection.pre.num_neurons
        print(pre_num_neurons)
    else:
        raise ValueError
    post_num_neurons = connection.post.num_neurons

    if isinstance(network.learning, SpikeProp):
        model.signal[connection] = {}
        model.signal[connection]['spike_response'] = np.zeros(shape=pre_num_neurons* network.learning.terminals)
        model.signal[connection]['synaptic_weight'] = np.ones(shape=(post_num_neurons, pre_num_neurons *network.learning.terminals))

        if connection.post == network.population[-1]:
            model.signal[connection]['synaptic_weight'].fill(0.005)
        elif connection.post != network.population[-1]:
            model.signal[connection]['synaptic_weight'].fill(0.0004)



        model.signal[connection]['delay']= np.zeros((pre_num_neurons, network.learning.terminals))

        for i in range (network.learning.terminals):
            model.signal[connection]['delay'][:,i]= i+1
        # print(model.signal[connection.pre]['spike_time'].shape)
        # print(model.signal[connection.post]['tau'])

        model.add_op(SpikeResponse_Spikeprop(current_period= model.signal['current_period'],
                                             pre_firing_time= model.signal[connection.pre]['spike_time'],
                                             spike_response= model.signal[connection]['spike_response'],
                                             tau= np.array(model.signal[connection.post]['tau']),
                                             num_terminals=np.array(network.learning.terminals),
                                             delays= model.signal[connection]['delay']))

        if connection.post == network.population[-1]:
            model.add_op(Inhibitory_neuron_effected(spike_response= model.signal[connection]['spike_response'],
                                                num_terminals= np.array(network.learning.terminals) ))     # Change sign of PSP of the first neuron


        model.add_op(MatMul(weight_matrix=model.signal[connection]['synaptic_weight'],
                            inp_vector=model.signal[connection]['spike_response'],
                            out_vector=model.signal[connection.post]['membrane_potential']))







def build_lif_population(model, obj):
    pass




def build_spikeprop(model, network, sampling_period):
    learning = network.learning

    model.signal[learning] = {}

    model.signal[learning]['prediction'] = model.signal[network.population[-1]]['spike_time']
    model.signal[learning]['target'] = np.zeros(shape=network.population[-1].num_neurons)
    model.signal[learning]['error'] = np.array([0.0])
    for i in range (len(network.population)):
        model.signal[network.population[i]]['upstream_gradient'] = np.zeros(shape= network.population[i].num_neurons)

    for j in range (len(network.connection)):
        model.signal[network.connection[j]]['derivative'] = np.zeros(shape= model.signal[network.connection[j]]['synaptic_weight'].shape)
        print('derivatives shape is ' + str(model.signal[network.connection[j]]['derivative'].shape))





    model.add_op(Update_target_firing_time(label= model.signal[network.source[0]]['label'],
                                           target= model.signal[learning]['target'],
                                           current_period= model.signal['current_period'],
                                           sampling_period= sampling_period,
                                           desired_firing_time=10.,
                                           non_desired_firing_time=20.))

    model.add_op(Update_error(error= model.signal[learning]['error'],
                              actual= model.signal[learning]['prediction'],
                              target= model.signal[learning]['target'],
                              sampling_period = sampling_period,
                              current_period= model.signal['current_period']))

    model.add_op(Compute_output_gradient(output_layer_firing_times= model.signal[network.population[-1]]['spike_time'],
                                         pre_layer_firing_times= model.signal[network.population[-2]]['spike_time'],
                                         delays=model.signal[network.connection[-1]]['delay'],
                                         target_firing_times=model.signal[learning]['target'],
                                         weights= model.signal[network.connection[-1]]['synaptic_weight'],
                                         tau= model.signal[network.population[-1]]['tau'],
                                         output_upstream_derivatives=  model.signal[network.population[-1]]['upstream_gradient'],
                                         current_period= model.signal['current_period'],
                                         sampling_period= sampling_period))

    model.add_op(Compute_hidden_gradient(output_layer_firing_times= model.signal[network.population[-1]]['spike_time'],
                                         hidden_layer_firing_time= model.signal[network.population[-2]]['spike_time'],
                                         input_layer_firing_times= model.signal[network.source[0]]['spike_time'],
                                         pre_delays= model.signal[network.connection[-2]]['delay'],
                                         post_delays= model.signal[network.connection[-1]]['delay'],
                                         pre_weights= model.signal[network.connection[-2]]['synaptic_weight'],
                                         post_weights= model.signal[network.connection[-1]]['synaptic_weight'],
                                         hidden_upstream_gradient=  model.signal[network.population[-2]]['upstream_gradient'],
                                         output_upstream_gradient= model.signal[network.population[-1]]['upstream_gradient'],
                                         tau= model.signal[network.population[-2]]['tau'],
                                         current_period=model.signal['current_period'],
                                         sampling_period=sampling_period
                                         ))
    model.add_op(Compute_output_derivaties(output_layer_firing_times=model.signal[network.population[-1]]['spike_time'],
                                           hidden_layer_firing_times=model.signal[network.population[-2]]['spike_time'],
                                           post_delays= model.signal[network.connection[-1]]['delay'],
                                           output_upstream_gradient=model.signal[network.population[-1]]['upstream_gradient'],
                                           tau= model.signal[network.population[-1]]['tau'],
                                           output_derivaties=model.signal[network.connection[-1]]['derivative'],
                                           current_period=model.signal['current_period'],
                                           sampling_period=sampling_period,
                                           learning_rate= np.array(network.learning.lr)
                                           ))

    model.add_op(Compute_hidden_derivaties(hidden_layer_firing_times=model.signal[network.population[-2]]['spike_time'],
                                           input_layer_firing_times=model.signal[network.source[0]]['spike_time'],
                                           pre_delays= model.signal[network.connection[-2]]['delay'],
                                           hidden_upstream_gradient=model.signal[network.population[-2]]['upstream_gradient'],
                                           tau= model.signal[network.population[-2]]['tau'],
                                           hidden_derivaties=model.signal[network.connection[-2]]['derivative'],
                                           current_period=model.signal['current_period'],
                                           sampling_period=sampling_period,
                                           learning_rate= np.array(network.learning.lr)
                                           ))
    model.add_op(Update_weights(weights=model.signal[network.connection[-1]]['synaptic_weight'],
                                derivaties= model.signal[network.connection[-1]]['derivative'],
                                current_period=model.signal['current_period'],
                                sampling_period=sampling_period,
                                ))

    model.add_op(Update_weights(weights=model.signal[network.connection[-2]]['synaptic_weight'],
                                derivaties= model.signal[network.connection[-2]]['derivative'],
                                current_period=model.signal['current_period'],
                                sampling_period=sampling_period,
                                ))



    pass

