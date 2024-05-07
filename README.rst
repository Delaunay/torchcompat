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


.. code-block:: text

   pip install torchcompat


Features
--------

* Provide a super set implementation of pytorch device interface
  to enable code to run seamlessly between different accelerators.
* Identify uniquely devices


.. code-block:: python

   import torchcompat.core as accelerator

   # on  cuda accelerator == torch.cuda
   # on  rocm accelerator == torch.cuda
   # on   xpu accelerator == torch.xpu
   # on gaudi accelerator == ...

   assert accelerator.is_available() == true
   assert accelerator.device_name in ('xpu', 'cuda', "hpu")           # rocm is seen as cuda by pytorch
   assert accelerator.device_string(0) == "cuda:0" or "xpu:0" or "hpu:0"
   assert accelerator.fetch_device(0) == torch.device("cuda:0")


   accelerator.set_enable_tf32(true) # toggle the right flags for each backend


Example
-------

.. code-block:: python

   example here
