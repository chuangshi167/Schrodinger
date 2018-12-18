#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger` package."""


import unittest
import tensorflow as tf
import argparse
import schrodinger.schrodinger as schrodinger
import math
tf.enable_eager_execution()

class TestSchrodinger(unittest.TestCase):

	def test_open_file(self):
		position, potential = schrodinger.open_file('potential_energy.dat')
		self.assertEqual(position, [0.0, 1.57079, 3.14159, 4.71238, 6.28318, 7.85398, 9.42477])
		self.assertEqual(potential, [0.0, 6.0, 0.0, -6.0, 0.0, 6.0, 0.0])
	

	def test_term(self):
		term_one = schrodinger.term(0)
		self.assertEqual(1, term_one(0).numpy())
		term_two = schrodinger.term(1)
		self.assertEqual(0, term_two(0).numpy())


	def test_create_basis(self):
		basis = schrodinger.create_basis(3)
		a = basis[0](0).numpy()
		b = math.cos(0)
		self.assertEqual(a, b)
		c = basis[1](math.pi/2).numpy()
		d = math.sin(math.pi/2)
		self.assertEqual(c, d)


	def test_v0(self):
		position = [0.0, 1.57079, 3.14159, 4.71238, 6.28318, 7.85398, 9.42477]
		potential = [0.0, 6.0, 0.0, -6.0, 0.0, 6.0, 0.0]
		position = tf.constant(position, shape = [1, len(position)], dtype = tf.float32)
		potential = tf.constant(potential, shape = [1, len(potential)], dtype = tf.float32)
		basis = schrodinger.create_basis(5)
		a = schrodinger.v0(position, potential, basis).numpy()
		b = [[ 6.0000000e+00], [ 1.8000000e+01], [ 1.0135167e-04], [-1.4448209e-05], [-6.0000000e+00]]
		self.assertEqual(a[0], b[0])
		self.assertEqual(a[1], b[1])
	

	def test_coefficient(self):
		position = [0.0, 1.57079, 3.14159, 4.71238, 6.28318, 7.85398, 9.42477]
		potential = [0.0, 6.0, 0.0, -6.0, 0.0, 6.0, 0.0]
		position = tf.constant(position, shape = [1, len(position)], dtype = tf.float32)
		potential = tf.constant(potential, shape = [1, len(potential)], dtype = tf.float32)
		basis = schrodinger.create_basis(5)
		coeff = schrodinger.coefficient(position, basis)
		self.assertEqual(coeff.get_shape(), [len(basis), len(basis)])
	

	def test_H_hat(self):
		position = [0.0, 1.57079, 3.14159, 4.71238, 6.28318, 7.85398, 9.42477]
		potential = [0.0, 6.0, 0.0, -6.0, 0.0, 6.0, 0.0]
		c = 1
		position = tf.constant(position, shape = [1, len(position)], dtype = tf.float32)
		potential = tf.constant(potential, shape = [1, len(potential)], dtype = tf.float32)
		basis = schrodinger.create_basis(5)
		v = schrodinger.v0(position, potential, basis)
		coeff = schrodinger.coefficient(position, basis)
		v0_hat = tf.linalg.solve(coeff, v)
		H = schrodinger.H_hat(c, len(basis), v0_hat)
		self.assertEqual(coeff.get_shape(), [len(basis), len(basis)])









		

if __name__ == '__main__':
    unittest.main()