#!/usr/bin/env python 
import mnist_loader
import network
#from timeit import Timer

#t = Timer("mnist_loader.load_data_wrapper()")
mnist_loader.load_data_wrapper()
#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
