
from n3ml.n3ml.Model import Model
from n3ml.n3ml.Builder import Builder
from n3ml.n3ml.BuilderDir.Spikeprob_Builder import Builder


def transpose(x, index):
    y = [None for _ in range(len(x))]
    for i, v in enumerate(index):
        y[i] = x[v]
    return y


class Simulator:
    def __init__(self,
                 network,
                 sampling_period,
                 model=None,
                 time_step=0.001,
                 ):
        self.network = network
        self.time_step = time_step
        self.sampling_period = sampling_period

        if model is None:
            self.model = Model()
            Builder.build(self.model, network, self.sampling_period)
            for i in range (len(self.model.operator)):
                print(i, self.model.operator[i])

            print(len(self.model.operator))


            # breakpoint()


    def run(self, simulation_time):
        import time
        import numpy as np
        import matplotlib.pyplot as plt

        num_steps = int(simulation_time / self.time_step)

        # ops for spikeprop
        # _ops = self.model.operator[:13]
        # ops = transpose(_ops, [4, 5, 7, 2, 3, 9, 10, 6, 11, 12, 8, 0, 1])

        '''Define blocks operators'''
        operator_List = self.model.operator[0:23]
        # operator_List.pop(14)


        operator_List = transpose(operator_List, [2,3,4,5, 6,8, 10, 11, 7, 12, 13, 14, 9, 15,16, 17,18,19,20,21, 22, 0,1])


        for step in range(num_steps):
            self._run_step(operator_List)
            '''Thisis function is used to visualize encoding '''
            # visualize.plotting(self.model.signal[self.network.source[0]]['image'], _ops[3](), current_period )

            print('label of this image is ' + str(self.model.signal[self.network.source[0]]['label']))
            print('spike-time of hidden-layer is ' + str(self.model.signal[self.network.population[0]]['spike_time']))
            print('spike-time of output-layer is ' + str(self.model.signal[self.network.population[1]]['spike_time']))

            if step == self.sampling_period:
                print('target firing_time is ' + str(self.model.signal[self.network.learning]['target']))
                print('error is ' + str(self.model.signal[self.network.learning]['error']))
                print('output_upstream_derivaties is ' + str(self.model.signal[self.network.population[1]]['upstream_gradient']))
                print('hidden_upstream_derivaties is ' + str(
                    self.model.signal[self.network.population[0]]['upstream_gradient']))

            print('next-period is ' + str(self.model.signal['current_period']))


            print('=========')



    def _run_step(self, ops):
        for op in ops:
            op()
