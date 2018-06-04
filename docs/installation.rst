Installation
============

The stellar population models in Bagpipes mean it is too large (~200 MB) to be hosted by PyPI. Instead, the packaged source for the most recent version can be downloaded here: `bagpipes-0.2.1 <https://drive.google.com/file/d/130PqprLfc07aRwI8SvrUQ7Sam7fWM--F/view?usp=sharing>`_.

Once you've downloaded and unzipped the source, in the main directory run:

.. code::

	python setup.py install

If you run into any issues with permissions, try:

.. code::

	python setup.py --user install

If that doesn't work either, you can simply add the top level bagpipes directory to your PYTHONPATH variable, e.g.

.. code::

	export PYTHONPATH="/Users/adam/bagpipes:$PYTHONPATH"

Though in this case you will need to manually install the Python package dependencies. 

If you have any trouble let me know.