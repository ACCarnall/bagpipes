Installation
============

The Bagpipes code and Python dependencies
-----------------------------------------

The packaged source for the most recent Bagpipes version can be downloaded at this link: `bagpipes-0.1.1 <http://dl.dropboxusercontent.com/s/lp2yef3xksrx6vw/bagpipes-0.1.1.tar.gz?dl=0>`_.

Once you've downloaded and unzipped the source, in the main directory run

.. code::

	python setup.py install

You can then delete the folder you downloaded and installed. You should now be able to import Bagpipes within Python (you'll receive a warning if you do not have MultiNest installed), let me know if this does not work!


Installing MultiNest
--------------------

If you want to perform model fitting (not model generation) you will also need to downlaod the MultiNest algorithm. To install MultiNest, clone  `github.com/JohannesBuchner/MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ and follow the instructions in the readme file. 

This can be a little tricky, I've found that on a mac, cmake often cannot find a fortran compiler. If this is the case, you should use homebrew to install one (note this will not work if you also have macports installed), and then tell cmake where to find your compiler. The whole process is:

.. code::

	git clone https://github.com/JohannesBuchner/MultiNest
	brew install gcc49
	export DYLD_LIBRARY_PATH="/usr/local/bin/gcc-4.9:$DYLD_LIBRARY_PATH"
	cd MultiNest/build
	cmake ..
	make
	sudo make install
	cd ../..
	rm -r MultiNest

If you run into trouble, further advice can be found `here <http://johannesbuchner.github.io/pymultinest-tutorial/install.html#on-your-own-computer>`_. I'm actively working on converting to a pure-python sampler to circumvent this step!

