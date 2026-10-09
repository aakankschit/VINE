Installation
============

VINE can be installed using pip or from source for development.

Requirements
------------

VINE requires Python 3.8 or later and the following dependencies:

* numpy
* scipy
* matplotlib
* pandas
* networkx
* torch
* pyro-ppl
* pydantic

Standard Installation
---------------------

Install VINE from PyPI:

.. code-block:: bash

   pip install vine-fep

Development Installation
------------------------

For development or to use the latest features:

.. code-block:: bash

   git clone https://github.com/aakankschit/VINE.git
   cd VINE
   pip install -e .

This installs VINE in development mode, allowing you to modify the source code and see changes immediately.

Optional Dependencies
---------------------

For development and testing:

.. code-block:: bash

   pip install vine-fep[dev]

This includes additional packages for testing, linting, and documentation:

* pytest
* pytest-cov
* black
* flake8
* mypy

For documentation building:

.. code-block:: bash

   pip install vine-fep[docs]

This includes:

* sphinx
* sphinx-rtd-theme
* myst-parser

Verification
------------

To verify your installation, run:

.. code-block:: python

   import vine_fep
   print(vine_fep.__version__)

You should see the version number printed without any errors.

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**PyTorch Installation**: If you encounter issues with PyTorch, install it separately first:

.. code-block:: bash

   pip install torch

**Pyro Installation**: For Pyro installation issues:

.. code-block:: bash

   pip install pyro-ppl

**Missing Dependencies**: If you get import errors, install missing dependencies:

.. code-block:: bash

   pip install numpy scipy matplotlib pandas networkx

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

For the best development experience, we recommend:

1. Using a virtual environment:

   .. code-block:: bash

      python -m venv vine-env
      source vine-env/bin/activate  # On Windows: vine-env\Scripts\activate
      pip install -e .[dev]

2. Setting up pre-commit hooks (if contributing):

   .. code-block:: bash

      pip install pre-commit
      pre-commit install
