EDGES analysis
==============

Re-analysis of the `Bowman et al. (2018) <https://www.nature.com/articles/nature25792>`_ results.

To use the package files, we recommend the following (after cloning to ``/users/johndoe/``)

.. code-block:: python

  import site
  site.addsitedir('/users/johndoe/edges/')

  from edges import data, utils, models


To run the ``global_sampler.py``, run

.. code-block:: python

  python edges/global_sampler.py -c <config_file>
