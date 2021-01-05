import numpy as np
np.set_printoptions(threshold=np.inf)

class Operator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError




'''======================================================================
   =============Time-step and Period Update Operators====================
   ======================================================================'''

class UpdateTime(Operator):
    def __init__(self, current_time):
        self.current_time = current_time

    def __call__(self, *args, **kwargs):
        self.current_time += 1

        return self.current_time

class UpdatePeriod(Operator):
    def __init__(self,
                 current_period,
                 sampling_period):
        self.current_period = current_period
        self.sampling_period = sampling_period

    def __call__(self, *args, **kwargs):
        if self.current_period < self.sampling_period:
            self.current_period += 1
            return self.current_period
        else:
            self.current_period.fill(0)
            return self.current_period



'''======================================================================
   ======================= Encoding Operators============================
   ======================================================================'''

class SampleLabel(Operator):
    def __init__(self,
                 label,
                 label_index,
                 current_period,
                 labels):
        # signals
        self.label = label
        self.label_index = label_index
        self.current_period = current_period
        self.labels = labels

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            self.label_index.fill(self.label_index+1)
            self.label.fill(self.labels[self.label_index])
            print(self.label)
            return self.label

class SampleImage(Operator):
    def __init__(self,
                 image,
                 image_index,
                 current_period,
                 images):
        # signals
        self.image = image
        self.image_index = image_index
        self.current_period = current_period
        self.images = images

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            import numpy as np
            self.image_index.fill(self.image_index+1)
            self.image[:] = self.images[self.image_index]

class InitSpikeTime(Operator):
    def __init__(self,
                 spike_time,
                 current_period,
                 value):
        # Signals
        self.spike_time = spike_time
        self.current_period = current_period
        #
        self.value = value

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            self.spike_time.fill(self.value)
        # print('shape of input spke-time is ' + str(self.spike_time.shape))




class PopulationEncode(Operator):
    def __init__(self,
                 image,
                 spike_time,
                 num_neurons,
                 interval,
                 beta,
                 min_value,
                 max_value,
                 sampling_period):
        self.image = image
        self.rows, self.cols = image.shape
        self.spike_time = spike_time
        self.num_neurons = num_neurons
        self.interval = interval
        self.beta = beta
        self.min_value = min_value
        self.max_value = max_value
        self.sampling_period = sampling_period
        from scipy.stats import norm
        self.receptive_field = [norm.pdf for _ in range(self.num_neurons)]
        import numpy as np
        self.mean = [self.min_value+(2*i-3)/2*(self.max_value-self.min_value)/(self.num_neurons-2) for i in np.arange(1, self.num_neurons+1)]
        self.std = 1/self.beta*(self.max_value-self.min_value)/(self.num_neurons-2)
        self.max_pdf = norm.pdf(0, scale=self.std)

    def __call__(self, *args, **kwargs):
        flatten_image = self.image.flatten()
        for i in range(self.rows * self.cols):
            for j in range(self.num_neurons):
                self.spike_time[i, j] = self._transform_spike_time(
                    self.receptive_field[j](flatten_image[i], self.mean[j], self.std))
        self.spike_time[self.spike_time>= self.interval]= self.sampling_period+1


    def _transform_spike_time(self, response):
        max_spike_time = self.interval
        max_response = self.max_pdf
        spike_time = response * max_spike_time / max_response
        spike_time = spike_time - max_spike_time
        spike_time = spike_time * -1.0
        spike_time = round(spike_time)
        return spike_time



'''======================================================================
   ======================= Population Operators============================
   ======================================================================'''
class SpikeTime(Operator):
    def __init__(self,
                 membrane_potential,
                 spike_time,
                 threshold,
                 current_period):
        self.membrane_potential = membrane_potential
        self.spike_time = spike_time
        self.threshold = threshold
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        # spike_time 중에서 not-to-fire 상태인 변수에 대해서만 아래 것을 계산하면 된다.
        self.spike_time[(self.spike_time <0) & (self.membrane_potential > self.threshold)] = self.current_period
        self.membrane_potential[self.membrane_potential > self.threshold] = 0

'''======================================================================
   ======================= Connection Operators============================
   ======================================================================'''

class SpikeResponse_Spikeprop(Operator):
    def __init__(self,
                 current_period,
                 pre_firing_time,
                 spike_response,
                 tau,
                 num_terminals,
                 delays):
        self.current_period = current_period
        self.pre_firing_time = pre_firing_time
        self.spike_response = spike_response
        self.tau = tau
        self.num_terminals= num_terminals
        self.delays = delays

    def __call__(self, *args, **kwargs):
        self.spike_time = np.copy(self.pre_firing_time)
        self.spike_time = self.spike_time.flatten()
        self.spike_time = np.tile(self.spike_time, (self.num_terminals,1))
        self.spike_time = np.transpose(self.spike_time, (1,0))

        self._t = self.current_period -  self.spike_time - self.delays
        self.x= self._t/self.tau
        self.y = np.exp(1 - self._t/self.tau)
        self.y = self.x* self.y
        self.y[(self.spike_time < 0) | (self.y < 0)] = 0
        # print(y.flatten().shape)
        self.spike_response[:] = self.y.flatten()


class Inhibitory_neuron_effected(Operator):
    def __init__(self, spike_response, num_terminals):
        self.spike_response = spike_response
        self.num_terminals = num_terminals

    def __call__(self, *args, **kwargs):
        import numpy as np
        self.spike_response[0: self.num_terminals] *=-1
        # print(self.spike_response)
        # print(self.out_vector)




class MatMul(Operator):
    def __init__(self, weight_matrix, inp_vector, out_vector):
        self.weight_matrix = weight_matrix
        self.inp_vector = inp_vector
        self.out_vector = out_vector

    def __call__(self, *args, **kwargs):
        import numpy as np
        self.out_vector[:] = np.matmul(self.weight_matrix, self.inp_vector)
        print(self.out_vector)
        # print(self.out_vector)


class Update_target_firing_time(Operator):
    def __init__(self, label, target, current_period, sampling_period, desired_firing_time, non_desired_firing_time ):
        self.label = label
        self.target = target
        self.current_period = current_period
        self.sampling_period = sampling_period
        self.desired = desired_firing_time
        self.non_desired = non_desired_firing_time

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            print(self.label)
            self.target.fill(self.non_desired)
            self.target[self.label] = self.desired

class Update_error(Operator):
    def __init__(self, error, actual, target, sampling_period, current_period ):
        self.error = error
        self.target = target
        self.actual = actual
        self.sampling_period = sampling_period
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.cals = self.target - self.actual
            self.cals = self.cals ** 2
            self.cals = np.sum(self.cals)
            self.cals /= 2.0
            self.error.fill(self.cals)

class Compute_output_gradient(Operator):
    def __init__(self, output_layer_firing_times, pre_layer_firing_times, delays, target_firing_times, weights, tau, output_upstream_derivatives, sampling_period, current_period):
        self.output_firing_times = output_layer_firing_times
        self.pre_layer_firing_times = pre_layer_firing_times
        self.delays = delays
        self.target_firing_times = target_firing_times
        self.weights = weights
        self.tau = tau
        self.output_upstream_derivaties = output_upstream_derivatives
        self.sampling_period = sampling_period
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.pre_layer_neurons, self.num_terminals = self.delays.shape
            self.output_layer_neurons = self.output_firing_times.shape[0]
            self.weights_copy = np.copy(self.weights)
            self.weights_copy = np.reshape(self.weights_copy, (self.output_layer_neurons, self.pre_layer_neurons, self.num_terminals))

            # print(self.delays.shape)
            # print(self.output_firing_times.shape)
            # print(self.weights.shape)
            # print(self.weights_copy.shape)

            for j in range (self.output_layer_neurons):
                self.numerator = self.target_firing_times[j] - self.output_firing_times[j]
                self.weighted_sum = 0.
                for i in range (self.pre_layer_neurons):
                    for l in range (self.num_terminals):
                        self.derivaties =0.
                        self._t = self.output_firing_times[j] - self.pre_layer_firing_times[i] - self.delays[i,l]
                        if self._t <=0:
                            self.derivaties =0.
                        elif self._t >0:
                            self.derivaties = np.exp(1- self._t/ self.tau)/ self.tau
                            self.derivaties = self.derivaties - (self._t*(np.exp(1- self._t/ self.tau)))/( self.tau**2)
                            if i ==0:
                                self.derivaties *= -1.

                        # print(i, j, self.derivaties)

                        self.weighted_sum += self.weights_copy[j,i,l] * self.derivaties
                if self.weighted_sum != 0.0:
                    self.output_upstream_derivaties[j] = self.numerator / self.weighted_sum
                else:
                    self.output_upstream_derivaties[j] = 0.0


class Compute_hidden_gradient(Operator):
    def __init__(self, output_layer_firing_times, hidden_layer_firing_time, input_layer_firing_times, pre_delays, post_delays, pre_weights, post_weights, hidden_upstream_gradient, tau, output_upstream_gradient, sampling_period, current_period):
        self.output_layer_firing_times = output_layer_firing_times
        self.hidden_layer_firing_time = hidden_layer_firing_time
        self.input_layer_firing_times = input_layer_firing_times
        self.pre_delays = pre_delays
        self.post_delays = post_delays
        self.pre_weights = pre_weights
        self.post_weights = post_weights
        self.hidden_upstream_gradient = hidden_upstream_gradient
        self.tau = tau
        self.output_upstream_gradient = output_upstream_gradient
        self.sampling_period = sampling_period
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.input_spike_time = np.copy(self.input_layer_firing_times)
            self.input_spike_time = self.input_spike_time.flatten()
            self.input_neurons = self.input_spike_time.shape[0]
            # print('input neurons is ' + str(self.input_neurons))
            self.hidden_neurons = self.hidden_layer_firing_time.shape[0]
            # print('hidden neurons is ' +str(self.hidden_neurons))
            self.output_neurons = self.output_layer_firing_times.shape[0]
            # print('output neurons is' + str(self.output_neurons))
            self.num_terminals = self.pre_delays.shape[1]
            # print(self.num_terminals)
            self.pre_weights_copy = np.copy(self.pre_weights)
            self.pre_weights_copy = np.reshape(self.pre_weights_copy, (self.hidden_neurons, self.input_neurons, self.num_terminals))
            self.post_weights_copy = np.copy(self.post_weights)
            self.post_weights_copy = np.reshape(self.post_weights_copy, (self.output_neurons, self.hidden_neurons, self.num_terminals))

            for i in range (self.hidden_neurons):
                self.numerator =0.0

                for j in range (self.output_neurons):
                    self.temp = 0.0

                    for k in range (self.num_terminals):
                       self.top_derivaties = 0.
                       self._t = self.output_layer_firing_times[j] - self.hidden_layer_firing_time[i] - self.post_delays[i, k]
                       if self._t <= 0:
                           self.top_derivaties = 0.
                       elif self._t > 0:
                           self.top_derivaties = np.exp(1 - self._t / self.tau) / self.tau
                           self.top_derivaties = self.top_derivaties - (self._t * (np.exp(1 - self._t / self.tau))) / (
                                   self.tau ** 2)
                           if i == 0:
                               self.top_derivaties *= -1.
                       self.temp += self.post_weights_copy[j,i,k] * self.top_derivaties

                    self.numerator += self.output_upstream_gradient[j] * self.temp

                self.denominator = 0.0

                for h in range (self.input_neurons):
                    for l in range (self.num_terminals):
                        self.bot_derivaties = 0.
                        self._t2 = self.hidden_layer_firing_time[i] - self.input_spike_time[h] - \
                                  self.pre_delays[h, l]
                        if self._t2 <= 0:
                            self.bot_derivaties = 0.
                        elif self._t2 > 0:
                            self.bot_derivaties = np.exp(1 - self._t2 / self.tau) / self.tau
                            self.bot_derivaties = self.bot_derivaties - (self._t2 * (np.exp(1 - self._t2 / self.tau))) / (
                                    self.tau ** 2)

                        self.denominator += self.pre_weights_copy[i,h,l] * self.bot_derivaties

                if self.denominator !=0.0:
                    self.hidden_upstream_gradient[i] = self.numerator/ self.denominator
                else:
                    self.hidden_upstream_gradient[i] = 0.0


class Compute_output_derivaties(Operator):
    def __init__(self, output_layer_firing_times, hidden_layer_firing_times, post_delays, output_upstream_gradient, tau, learning_rate, output_derivaties, sampling_period, current_period):
        self.output_layer_firing_times = output_layer_firing_times
        self.hidden_layer_firing_times = hidden_layer_firing_times
        self.post_delays = post_delays
        self.output_upstream_gradient = output_upstream_gradient
        self.tau = tau
        self.learning_rate = learning_rate
        self.output_derivaties = output_derivaties
        self.sampling_period = sampling_period
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:

            self.output_neurons = self.output_layer_firing_times.shape[0]
            self.hidden_neurons, self.num_terminals = self.post_delays.shape
            print(self.output_neurons, self.hidden_neurons, self.num_terminals)
            self.output_derivaties_copy = np.copy(self.output_derivaties)
            self.output_derivaties_copy = np.reshape (self.output_derivaties_copy, (self.output_neurons, self.hidden_neurons, self.num_terminals))

            for j in range (self.output_neurons):
                for i in range(self.hidden_neurons):
                    for k in range(self.num_terminals):
                        self.r =0
                        self._t = self.output_layer_firing_times[j] - self.hidden_layer_firing_times[i] - self.post_delays[i,k]
                        if self._t <=0:
                            self.r =0
                        elif self._t >0:
                            self.x = self._t /self.tau
                            self.r = np.exp(1 - self._t/self.tau)
                            self.r = self.x * self.r

                        self.output_derivaties_copy[j,i,k] = (-self.learning_rate) * self.r * self.output_upstream_gradient[j]

            self.output_derivaties_copy = np.reshape(self.output_derivaties_copy, (self.output_neurons, self.hidden_neurons * self.num_terminals))
            self.output_derivaties[:] = self.output_derivaties_copy


class Compute_hidden_derivaties(Operator):
    def __init__(self, hidden_layer_firing_times, input_layer_firing_times, pre_delays,
                 hidden_upstream_gradient, tau, learning_rate, hidden_derivaties, sampling_period,
                 current_period):
        self.input_layer_firing_times = input_layer_firing_times
        self.hidden_layer_firing_times = hidden_layer_firing_times
        self.pre_delays = pre_delays
        self.hidden_upstream_gradient = hidden_upstream_gradient
        self.tau = tau
        self.learning_rate = learning_rate
        self.hidden_derivaties = hidden_derivaties
        self.sampling_period = sampling_period
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:

            self.hidden_neurons = self.hidden_layer_firing_times.shape[0]
            self.input_neurons, self.num_terminals = self.pre_delays.shape
            print(self.hidden_neurons, self.input_neurons, self.num_terminals)
            self.input_spike_time = np.copy(self.input_layer_firing_times)
            self.input_spike_time = self.input_spike_time.flatten()

            self.hidden_derivaties_copy = np.copy(self.hidden_derivaties)
            self.hidden_derivaties_copy = np.reshape(self.hidden_derivaties_copy, (
            self.hidden_neurons, self.input_neurons, self.num_terminals))

            for j in range(self.hidden_neurons):
                for i in range(self.input_neurons):
                    for k in range(self.num_terminals):
                        self.r = 0
                        self._t = self.hidden_layer_firing_times[j] - self.input_spike_time[i] - \
                                  self.pre_delays[i, k]
                        if self._t <= 0:
                            self.r = 0
                        elif self._t > 0:
                            self.x = self._t / self.tau
                            self.r = np.exp(1 - self._t / self.tau)
                            self.r = self.x * self.r

                        self.hidden_derivaties_copy[j, i, k] = (-self.learning_rate) * self.r * \
                                                               self.hidden_upstream_gradient[j]

            self.hidden_derivaties_copy = np.reshape(self.hidden_derivaties_copy, (
            self.hidden_neurons, self.input_neurons * self.num_terminals))
            self.hidden_derivaties[:] = self.hidden_derivaties_copy


class Update_weights(Operator):
    def __init__(self, weights, derivaties, sampling_period, current_period):
        self.weights = weights
        self.derivaties = derivaties
        self.sampling_period = sampling_period
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.weights[:] = self.weights+ self.derivaties
            self.weights[self.weights < 0.0] = 0.0

        # print(self.out_vector)
