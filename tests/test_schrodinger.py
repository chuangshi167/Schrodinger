#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger` package."""


import unittest
import pytest
import tensorflow as tf
import schrodinger.schrodinger as schrodinger
import math

class TestSchrodinger(unittest.TestCase):
	# def test_create_args(self):
	# 	args = schrodinger.create_args()
	# 	self.assertEqual(args.file, 'potential_energy.dat')
	# 	self.assertEqual(args.c, 1.0)
	# 	self.assertEqual(args.size, 5)
		# self.assertEqual(args.domain, [0, 7])


	def test_open_file(self):
		position, potential = schrodinger.open_file('potential_energy.dat')
		self.assertEqual(position, [0.0, 1.57079, 3.14159, 4.71238, 6.28318, 7.85398, 9.42477])
		self.assertEqual(potential, [0.0, 6.0, 0.0, -6.0, 0.0, 6.0, 0.0])
	
	
	def test_term(self):
		function1 = schrodinger.term(0)
		function2 = lambda x: tf.math.cos(0 * x)
		self.assertTrue(tf.equal(function1(math.pi), function2(math.pi)))
	# def test_create_basis(self):


	def v0(position, potential, basis):
    	_product = tf.constant([], dtype = tf.float32)
    	for i in range(len(basis)):
        	matrix1 = basis[i](position)
        	matrix2 = tf.transpose(potential)
        	product = matrix1 * matrix2
        	product = tf.reduce_sum(product)
        	product = tf.constant(product, shape = [1,])
        	_product = tf.concat([_product, product], 0)
		return _product
	

	def v0_hat(position, potential, basis):
		

if __name__ == '__main__':
    unittest.main()