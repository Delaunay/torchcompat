torchcompat
=============================

|pypi| |py_versions| |codecov| |docs| |tests| |style|

.. |pypi| image:: https://img.shields.io/pypi/v/torchcompat.svg
    :target: https://pypi.python.org/pypi/torchcompat
    :alt: Current PyPi Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/torchcompat.svg
    :target: https://pypi.python.org/pypi/torchcompat
    :alt: Supported Python Versions

.. |codecov| image:: https://codecov.io/gh/Delaunay/torchcompat/branch/master/graph/badge.svg?token=40Cr8V87HI
   :target: https://codecov.io/gh/Delaunay/torchcompat

.. |docs| image:: https://readthedocs.org/projects/torchcompat/badge/?version=latest
   :target:  https://torchcompat.readthedocs.io/en/latest/?badge=latest

.. |tests| image:: https://github.com/Delaunay/torchcompat/actions/workflows/test.yml/badge.svg?branch=master
   :target: https://github.com/Delaunay/torchcompat/actions/workflows/test.yml

.. |style| image:: https://github.com/Delaunay/torchcompat/actions/workflows/style.yml/badge.svg?branch=master
   :target: https://github.com/Delaunay/torchcompat/actions/workflows/style.yml


.. code-block:: bash

   pip install torchcompat


Features
--------

Provide a super set implementation of pytorch device interface to enable code to run seamlessly between
different accelerators.

That means that torchcompat provide a superset of each implementations.

.. code-block::

   import torchcompat.core as accelerator

   # on  cuda accelerator == torch.cuda
   # on  rocm accelerator == torch.cuda
   # on   xpu accelerator == torch.xpu
   # on gaudi accelerator == ...

   assert accelerator.is_available() == true
