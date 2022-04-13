
Python-based Visualization of Field Computations from the MATLAB library "Optical Tweezers Toolbox"
===============================================================

Personally, I find dealing with MATLAB directly to be a pain
and I'm much more comfortable producing visualizations with 
Python. This collection of software is meant to make it easy to 
produce high-quality visualisations of the optical field that 
results from typical Mie scattering problems.


Introduction
------------

The physical description of the incident optical field and the 
subsequent scattering are all performed by MATLAB, making use of
the T-matrix formalism. See the link below for a complete 
description of the software together with accompanying literature.

`Optical Tweezers Toolbox <https://www.mathworks.com/matlabcentral/fileexchange/73541-ott-optical-tweezers-toolbox>`_ - Lenton, Isaac C. D., et al.


Install
-------

Non-Python prerequisites
````````````````````````

Users will need to `install the MATLAB engine for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_ specific 
to their personal work environment, as well as the ``unwrap`` package for 
two-dimensional phase unwrapping. Other than that, standard NumPy and 
Scipy installations should cover everything you neeed. The explicit
requirements will be enumerated soon.


From sources
````````````

To install system-wide, noting the path to the src since no wheels
exist on PyPI, use::

   pip install ./ott_visualization

If you intend to edit the code and want the import calls to reflect
those changes, install in developer mode::

   pip install -e ott_visualization

If you don't want a global installation (i.e. if multiple users will
engage with and/or edit this library) and you don't want to use venv
or some equivalent::

   pip install -e ott_visualization --user

where pip is pip3 for Python3 (tested on Python 3.6.9). Be careful 
NOT to use ``sudo``, as the latter two installations make a file
``easy-install.pth`` in either the global or the local directory
``lib/python3.X/site-packages/easy-install.pth``, and sudo will
mess up the permissions of this file such that uninstalling is very
complicated.


Uninstall
---------

If installed without ``sudo`` as instructed, uninstalling should be 
as easy as::

   pip uninstall ott_visualization

If installed using ``sudo`` and with the ``-e`` and ``--user`` flags, 
the above uninstall will encounter an error.

Navigate to the file ``lib/python3.X/site-packages/easy-install.pth``, 
located either at  ``/usr/local/`` or ``~/.local`` and ensure there
is no entry for ``opt_lev_analysis``.


License
-------

The package is distributed under an open license (see LICENSE file for
information).


Authors
-------

Charles Blakemore (chas.blakemore@gmail.com)