# -*- coding: utf-8 -*-

"""Main module."""

import argparse
import tensorflow as tf

def create_args():
	"""
	Read parameters from user input
	"""	
	parser = argparse.ArgumentParser(description = "Necessary input")
	parser.add_argument('--file', type = str, default = 'potential_energy.dat', help = 'potential energy file')
	parser.add_argument('--c', type = float, default = 1.0, help = 'constant c for the kinetic energy term')
	parser.add_argument('--size', type = int, default = 5, help = 'the size of the basis')
	# parser.add_argument('--domain', type = list, default = [0, 7], help = 'the domain of the function')
	args = parser.parse_args()
	return args

def open_file(filename):
	position = []
	potential = []
	file = open(filename, 'r')
	for line in file:
		if(line[0] != '#'):
			data = line.split()
			position.append(float(data[0]))
			potential.append(float(data[1]))
	file.close()
	return position, potential


def term(i):
	"""
	Create the ith term in the basis
	"""
	if (i % 2 == 0):
		return lambda x: tf.math.cos(i/2 * x)
	else:
		return lambda x: tf.math.sin((i + 1)/2 * x)

def create_basis(n):
    basis = []
    for i in range(n):
        basis.append(term(i))
    return basis





def main():
	position, potential = open_file('potential_energy.dat')
	potential = tf.constant(potential, dtype = tf.float32)
	position = tf.constant(position, dtype = tf.float32)


if __name__ == '__main__':
    main()
