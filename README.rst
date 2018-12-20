===========
Schrodinger
===========



.. image:: https://img.shields.io/travis/chuangshi167/Schrodinger.svg
        :target: https://travis-ci.org/chuangshi167/Schrodinger



.. image:: https://coveralls.io/repos/github/chuangshi167/Schrodinger/badge.svg?branch=master
        :target: https://coveralls.io/github/chuangshi167/Schrodinger?branch=master





Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT license
* Documentation: https://schrodinger.readthedocs.io.

Descriotion
-----------

This project is for CHE477, University of Rochester.
It solves the 1-D Schrodinger question of a particle in a given potential field by converting it to an Eigenvector-Eigenvalue question.

How to use
----------

This simulator can be invoked from the terminal, using the following command::

	python3 schrodinger/schrodinger.py
 
There are a few preset parameters that can be modified in the argument.

They are:

* -- file  
        * Type: string 
        * default = potential_energy.dat

* -- c
        * Type: float
        * default = 1.0

* -- size
        * Type: int
        * default = 5


If you would like to change any of the preset parameters, using the following command::

	python3 schrodinger/schrodinger.py --file potential_energy.dat --c 1.0 --size 5

How to run the tests
--------------------
The tests can be invoked from the terminal, using the following command::

	coverage run setup.py test


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
