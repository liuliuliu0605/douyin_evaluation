from sampling import *
from util import *

root = 1
iteration = 100
#file_path = r'G:\Sampling\dataset\Cit-HepPh.txt'
file_path = r'G:\Sampling\dataset\test.txt'

#graph = load_graph(file_path)
graph = load_network(file_path)

def example_sample():
    print(type(graph))
    sample = Sampling(iteration, G=graph)
    print(sample.get_friends(1))
    print(sample.get_degree(3))
    print(sample.get_friend_random(2))

def example(choice=0):
    if choice == 0:
        instance = BFS(iteration, G=graph)
    elif choice == 1:
        instance = RW(iteration, G=graph)
    else:
        instance = MHRW(iteration, G=graph)
    instance.run(root)
    samples = instance.get_samples()
    samples_duplicate = instance.get_samples_duplicate()
    avg_degrees = instance.avg_degrees()
    avg_degrees_duplicate = instance.avg_degrees_duplicate()
    print("Duplicate samples: ", samples_duplicate)
    print(len(samples_duplicate), avg_degrees_duplicate)
    print("Samples: ", samples)
    print(len(samples), avg_degrees)
    fig = instance.virtualization()
    fig.savefig('./test/network.png')

def test_util():
    graph_revised = revise_graph(graph,[3,4])
    print(type(graph_revised))
    print(graph.degree())
    print(graph_revised.degree())

#test_util()
example(0)
#example_sample()

