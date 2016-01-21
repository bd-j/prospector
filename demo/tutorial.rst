Tutorial
============

Here is a quick demo of how to get up and running with |Codename|.

We assume you have installed |Codename| and all its dependencies as laid out in the docs.
The next thing you need to do is make a temporary work directory, <workdir>

.. code-block:: shell
		
		cd <workdir>
		cp <codedir>/scripts/prospector.py .
		cp <codedir>/demo/demo_* .

We now have a prospector executable, a *parameter file*  or two, and some data.
Take a look at the data file in an editor, you'll see it its a simple ascii file, with a few rows and several columns.
Each row is a different galaxay, each column is a different piece of information about that galaxy.

This is just an example.
In practice |Codename| can work with a wide variety of data types.

Open up ``demo_params.py`` in an editor, preferably one with syntax highlighting.
You'll see that it's a python file.
Some things are imported, and then there is the ``run_params`` dictionary.
This dictionary is where you put variables that control the operation of the code.
It is passed to each of the other main setup functions in ``param_file.py``

About those imports.
Since we are fitting galaxies with a composite stellar population,
we made sure to import the source_basis.CSPBasis class.
If you were fitting stars or non-parameteric SFHs you would use a different object from source_basis.
We also made a little alias for a particular prior function, the tophat.

The next thing to look at is the ``load_obs()`` function.
This is where you take the data from whatever format you have and
put it into the format required by |Codename| for a single object.
This means you will have to modify this function heavily for your own use.
But it also means you can use your existing data formats.

Right now, the ``load_obs`` function just reads ascii data from a file,
picks out a row, and then makes a dictionary using data in that row.
You'll note that both the datafile name and the object number are keyword arguments to this function.
That means they can be set at execution time on the command line,
by also including those variables in the ``run_params`` dictionary.
We'll see an example later.

When you write your own ``load_obs`` function, you can add all sorts of keyword arguments that control its output.
You can also import helper functions and modules.
These can be either things like astropy, h5py, and sqlite or your own project specific modules and functions.
As long as the output dictionary is in the right format, the body of this function can do anything.

Ok, now we go to the ``load_sps`` function.
This one is pretty straightforward, it simply instantiates our CSPBasis object.
After that is ``load_gp``, ignore that for now.

Now on the fun part.
The ``model_params`` list is where the model that we will fit is specified.
Each entry in the list is a dictionary that describes a single parameter.
You'll note that for 5 of these parameters we have set ``"isfree": True``.
These are the parameters that will be varied during the fit.
We have set a prior on these parameters.
Other parameters have their value set (by the ``"init"`` key) but do not vary during the fit.
They can be made to vary by setting ``"isfree": True`` and specifying a prior.
Parameters not listed here will be set to their default values.
For CSPBasis this means the default values in the ``fsps.StellarPopulation()`` object,
see `python-fsps (http://dan.iel.fm/python-fsps/current/)`_ for details

Finally, the ``load_model()`` function takes the ``model_params`` list and
uses it to instantiate a ``SedModel`` object.
If you wanted to change the specification of the model using command line arguments,
you could do it in this function using keyword arguments that are also keys of ``run_params``.

Running a fit
----------------------

To run this fit on object 0, we would do the following at the command line

.. code-block:: shell
		
		python prospector.py --param_file=demo_params.py --objid=0 --outfile=demo_obj0

If we wanted to change something about the MCMC parameters, we could also do that at the command line

.. code-block:: shell
		
		python prospector.py --param_file=demo_params.py --objid=0 --outfile=demo_obj0 \
		--nwalkers=32 --niter=1024

Finally, it is sometimes useful to run the script from the interpreter to do some checks.
This is best done with the IPython enhanced interactive python.

.. code-block:: shell
		
		ipython
		In [1]: %run prospector.py --param_file=demo_params.py --objid=0 --debug=True

The ``--debug=True`` flag will halt execution just before the fitting starts.
You can then inspect the ``obsdat`` dictionary, the ``model`` object,
and the ``run_params`` dictionary to make sure everything is working fine.

Working with the output
--------------------------------
After the fit is completed we should have a number of files with names like
``demo_obj0_<timestamp>_*``.
We will read these in with python and make some plots using utilities in |Codename|

.. |Codename| replace:: BSFH
