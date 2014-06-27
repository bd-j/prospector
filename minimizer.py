import numpy as np
from scipy.optimize import minimize
try:
    import multiprocessing
except ImportError:
    pass

class Pminimize(object):
    """
    A class for performing minimizations using scipy's minimize
    function.  This class enables minimizations from different initial
    positions to be parallelized.

    If a pool object is passed, that objects map function is used to
    parallelize the minimizations.  Otherwise, if nthreads > 1, we
    generate a pool using multiprocessing and use that map function

    :param chi2:
        Function to be minimized.
        
    :param opts:
        Dictionary of options for the minimization, as described in
        scipy's minimize docs.

    :param model:
        A 'model' object, but really just an object that is passed as
        an argument to your chi2 function
        
    :param method:  (Default: 'powell')
        Minimization method.  This should probably be 'powell', since
        I haven't tried anything else and there's no way to pass
        hessians or jacobians.

    :param pool: (Default: None)
        A pool object which contains a map method that will be used
        for distributing tasks.

    :param nthreads: (Default: 1)
        If pool is None and nthreads > 1, create a mutiprocessing pool
        with nthreads threads.  Otherwise, this is ignored.
         
    """
    def __init__(self, chi2, opts, model, method='powell', pool=None, nthreads=1):
        self.method = method
        self.opts = opts
        self.model = model
        self._size = None

        # Wrap scipy's minimize to make it pickleable
        self.minimize = _function_wrapper(minimize, [chi2, model, method, opts])
        
        self.pool = pool
        if nthreads > 1 and self.pool is None:
            self.pool = multiprocessing.Pool(nthreads)
            self._size = nthreads
        
    def run(self, pinit):
        """
        Actually run the minimizations, in parallel if pool has been
        set up.

        :param pinit:
           An iterable where each element is a parameter vector for a
           starting condition.

        :returns results:
            A list of scipy-style minimization results objects,
            corresponding to the initial locations.
        """
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map

        results = list( M(self.minimize,  [np.array(p) for p in pinit]) )
        return results

    @property
    def size(self):
        if self.pool is None:
            return 1
        elif self._size is not None:
        #    print('uhoh')
            return self._size
        else:
            return self.pool.size
    

class _function_wrapper(object):
    """
    This is a hack to make the minimization function pickleable (for
    MPI) even though it requires many arguments.  Ripped off from emcee.
    """
    def __init__(self, f, args):
        self.f = f
        self.args = args

    def __call__(self, x):
        try:
            return self.f(self.args[0], x, args = (self.args[1],),
                          method = self.args[2], options = self.args[3])
        except:
            import traceback
            print("minimizer: Exception while trying to minimize the function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  exception:")
            traceback.print_exc()
            raise
