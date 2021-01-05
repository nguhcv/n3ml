from n3ml.n3ml.Network import Network
from n3ml.n3ml.Source import MNISTSource, MNISTSource2
from n3ml.n3ml.Connection import Connection
from n3ml.n3ml.Population import SRMPopulation
from n3ml.n3ml.Learning import SpikeProp
from n3ml.n3ml.Simulator import Simulator

if __name__ == '__main__':
    net = Network(code='single', learning=SpikeProp(lr=0.0075), threshold=1.)

    src = MNISTSource(code='population', num_neurons=8, sampling_period=10)           #interval in ms

    pop_1 = SRMPopulation(num_neurons=20, tauLeak=7)
    pop_2 = SRMPopulation(num_neurons=10, tauLeak=7)

    conn_1 = Connection(pre=src, post=pop_1)
    conn_2 = Connection(pre=pop_1, post=pop_2)

    net.add(src)
    net.add(conn_1)
    net.add(pop_1)
    net.add(conn_2)
    net.add(pop_2)

    sim = Simulator(network=net, sampling_period=23)


    sim.run(simulation_time=0.05)
