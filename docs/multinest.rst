
Optional: Installing MultiNest
------------------------------

The default sampler in Bagpipes is now Dynesty, which is pure-Python and should be installed as part of the process above. Installing MultiNest is therefore no longer necessary to use the code.

Fitting with MultiNest is however still supported, if you want to use it, first clone  `github.com/JohannesBuchner/MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ and follow the instructions in the readme file. 

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

