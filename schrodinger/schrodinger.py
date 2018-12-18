# -*- coding: utf-8 -*-

"""Main module."""

import argparse
import tensorflow as tf
tf.enable_eager_execution()

def create_args():
	"""
	Read parameters from user input
	"""	
	parser = argparse.ArgumentParser(description = "Necessary input")
	parser.add_argument('--file', type = str, default = 'potential_energy.dat', help = 'potential energy file')
	parser.add_argument('--c', type = float, default = 1.0, help = 'constant c for the kinetic energy term')
	parser.add_argument('--size', type = int, default = 5, help = 'the size of the basis')
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
	if i % 2 == 0:
		return lambda x: tf.math.cos((i/2) * x)
	else:
		return lambda x: tf.math.sin((i + 1)/2 * x)


def create_basis(n):
    basis = []
    for i in range(n):
        basis.append(term(i))
    return basis


def v0(position, potential, basis):
    matrix1 = basis[0](position)
    matrix2 = potential
    _product = matrix1 * matrix2
    _product = tf.math.reduce_sum(_product, -1)
    _product = tf.reshape(_product, [1, 1])
    for i in range(1, len(basis)):
        matrix1 = basis[i](position)
        product = matrix1 * matrix2
        product = tf.math.reduce_sum(product, -1)
        product = tf.reshape(product, [1, 1])
        _product = tf.concat([_product, product], -1)
    _product = tf.reshape(_product, [len(basis), 1])
    return _product


def coefficient(position, basis):
    sub_coefficient_matrix = basis[0](position)
    sub_coefficient_matrix = tf.reshape(sub_coefficient_matrix, [position.get_shape()[1], 1])
    
    for i in range(1, len(basis)):
        matrix1 = basis[i](position)
        matrix1 = tf.reshape(matrix1, [position.get_shape()[1], 1])
        sub_coefficient_matrix = tf.concat([sub_coefficient_matrix, matrix1], -1)


    matrix2 = basis[0](position)
    matrix2 = tf.reshape(matrix2, shape = [7,1])
    coefficient_matrix = sub_coefficient_matrix * matrix2
    coefficient_matrix = tf.math.reduce_sum(coefficient_matrix, 0)
    coefficient_matrix = tf.reshape(coefficient_matrix, [1, coefficient_matrix.get_shape()[0]])
    
    for j in range(1, len(basis)):
        matrix2 = basis[j](position)
        matrix2 = tf.reshape(matrix2, shape = [position.get_shape()[1],1])
        trial = sub_coefficient_matrix * matrix2
        trial = tf.math.reduce_sum(trial, 0)
        trial = tf.reshape(trial, [1, trial.get_shape()[0]])
        coefficient_matrix = tf.concat([coefficient_matrix, trial], 0)
    return coefficient_matrix


def H_hat(c, n, v0_hat):
    matrix = tf.zeros([n, 1])
    for i in range(1, n):
        column = 0
        for j in range(0, n):
            if j == i:
                element = tf.constant([((i + 1)//2) ** 2], shape = [1, 1], dtype = tf.float32)
            else:
                element = tf.constant([0], shape = [1, 1], dtype = tf.float32)
            if column == 0:
                column = element
            else:
                column = tf.concat([column, element], 0)
        matrix = tf.concat([matrix, column], -1)
    matrix = matrix * c
    matrix2 = v0_hat
    for i in range(1, n):
        matrix2 = tf.concat([matrix2, v0_hat], -1)
    H = matrix + matrix2
    return H


def main():
	args = create_args()
	c = args.c
	n = args.size
	file = args.file
	position, potential = open_file(file)
	position = tf.constant(position, shape = [1, len(position)], dtype = tf.float32)
	potential = tf.constant(potential, shape = [1, len(potential)], dtype = tf.float32)
	basis = create_basis(n)
	v = v0(position, potential, basis)
	coeff = coefficient(position, basis)
	v0_hat = tf.linalg.solve(coeff, v)
	H = H_hat(c, len(basis), v0_hat)
	e, v = tf.linalg.eigh(H)
	print(e[0].numpy())
	print(v[0].numpy())


if __name__ == '__main__':
    main()
