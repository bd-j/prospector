import numpy as np
import pyfits
import scipy.spatial
import astropy.constants as constants

#import observate

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
hplanck = constants.h.cgs.value
c_cgs = constants.c.cgs.value
kboltz = constants.k_B.cgs.value

class ModelLibrary(object):
    """Class to deal with (irregular) grids of models.  primary attribute
    is pars: a structured array of parameter values of shape (ngrid)
    """
    
    def __init__(self,pars=None, parnames = None):
        if pars is not None :
            self.set_pars(pars, parnames)
            
    def set_pars(self, pars,parnames):
        self.pars = self.structure_array(pars,parnames)
        self.ngrid = self.pars.shape[0]

    def add_par(self,value,name, dtype='<f8'):
        self.pars = self.join_struct_arrays( [self.pars, self.structure_array(value, name, dtype = dtype)] )
        pass

    def par_names(self):
        return self.pars.dtype.names

    def par_range(self, parname, inds=None):
        prange=[]
        for p in parname:
            range_list = [ np.nanmin(self.pars[inds][p]),np.nanmax(self.pars[inds][p]) ]
            prange.append(range_list)
        return prange

    def structure_array(self, values,fieldnames, types=['<f8']):
        """turn a numpy array of floats into a structurd array. fieldnames can be a list or
        string array of parameter names with length nfield.
        Assumes pars is a numpy array of shape (nobj,nfield)
        """
        values=np.atleast_2d(values)
        if values.shape[-1] != len(fieldnames):
            if values.shape[0] == len(fieldnames):
                values=values.T
            else:
                raise ValueError('ModelLibrary.structure_array: array and fieldnames do not have consistent shapes!')
        nobj=values.shape[0]
        
        #set up the list of tuples describing the fields.  Assume each parameter is a float
        fieldtuple=[]
        for i,f in enumerate(fieldnames):
            if len(types) > 1 :
                tt =types[i]
            else: tt=types[0]
            fieldtuple.append((f,tt))
        #create the dtype and structured array                    
        dt=np.dtype(fieldtuple)
        struct=np.zeros(nobj,dtype=dt)
        for i,f in enumerate(fieldnames):
            struct[f]=values[...,i]
        return struct

    def join_struct_arrays(self, arrays):
        """from some dudes on StackOverflow.  add equal length
        structured arrays to produce a single structure with fields
        from both.  input is a sequence of arrays."""
        if False in [len(a) == len(arrays[0]) for a in arrays] :
            raise ValueError ('join_struct_arrays: array lengths do not match.')

        newdtype = np.dtype(sum((a.dtype.descr for a in arrays), []))        
        if len(np.unique(newdtype.names)) != len(newdtype.names):
            raise ValueError ('join_struct_arrays: arrays have duplicate fields.')
        newrecarray = np.empty(len(arrays[0]), dtype = newdtype)
        for a in arrays:
            for name in a.dtype.names:
                newrecarray[name] = a[name]
        return newrecarray

    def model_weights(self, target_points, parnames=None, itype='dt',subinds=None):
        """Given an ndarray of target points of shape (ntarg,ndim) and optionally a
        list of parnames of length (ndim) construct the corresponding model grid ndarry,
        pass to weightsDT, and return the indices and weights of the model grid points
        corresponding to linear intrpolation to the target points"""
        
        #deal with recarray input
        if parnames is None:
            parnames = target_points.dtype.names #if the target point(s) is already a struct array use the field names
            targets = np.array(target_points.tolist())
        else:
            targets = target_points
        targets=np.atleast_2d(targets)
            
        #pull the grid points out of the model record data and make an (nmodel,ndim) array of
        #model grid parameter values.  need to loop to make sure order is correct
        model_points=[]
        for pname in parnames:
            model_points.append(np.squeeze(self.pars[subinds][pname]))
        model_points = np.array(model_points).transpose() #(nmod, ndim)

        #pass the result to weightsDT
        return self.weightsDT(model_points,targets)

    def weightsDT(self,model_points,target_points):
        """ The interpolation weights are determined from barycenter coordinates
        of the vertices of the enclosing Delaunay triangulation. This allows for
        the use of irregular Nd grids. see also weightsLinear.
            model_points - array of shape (nmod, ndim)
            target_points - array of shape (ntarg,ndim)
            output inds and weights - arrays of shape (npts,ndim+1)
        """
        
        ndim = target_points.shape[-1]
        #delaunay triangulate and find the encompassing (hyper)triangle(s) for the desired point
        dtri = scipy.spatial.Delaunay(model_points)
        #output triangle_inds is an (ntarg) array of simplex indices
        triangle_inds = dtri.find_simplex(target_points)
        #and get model indices of (hyper)triangle vertices. inds has shape (ntarg,ndim+1)
        inds = dtri.vertices[triangle_inds,:]
        #get the barycenter coordinates through matrix multiplication and dimensional juggling
        bary = np.dot( dtri.transform[triangle_inds,:ndim,:ndim],
                       (target_points-dtri.transform[triangle_inds,ndim,:]).reshape(-1,ndim,1) )
        oned = np.arange(triangle_inds.shape[0])
        bary = np.atleast_2d(np.squeeze(bary[oned,:,oned,:])) #ok.  in np 1.7 can add an axis to squeeze
        last = 1-bary.sum(axis=-1) #the last bary coordinate is 1-sum of the other coordinates
        weights = np.hstack((bary,last[:,np.newaxis]))

        #loop implementation of the above for clarity
        #npts = triangle_inds.shape[0]
        #bary = np.zeros([npts,ndim+1])
        #for i in xrange(npts):
        #    bary[i,:-1]= np.dot( dtri.transform[triangle_inds[0],:ndim,:ndim],
        #                       (target_points-dtri.transform[triangle_inds[i],ndim,:])
            
        return inds, weights

    def weights_1DLinear(self, model_points, target_points):
        order = model_points.argsort()
        mod_sorted = model_points[order]
        ind_nearest = np.searchsorted(mod_sorted, target_points,side='left')

        maxind = mod_sorted.shape[0]-1
        edge = np.logical_or( ind_nearest == 0, ind_nearest == (maxind+1) )
        inds = (np.vstack([order[np.clip(ind_nearest, 0, maxind)], order[np.clip(ind_nearest-1, 0,maxind)]])).T

        d1 = np.absolute( model_points[inds[:,0]] - target_points )
        d2 = np.absolute( model_points[inds[:,1]] - target_points )
        width = d1+d2
        width[edge] = 1
        weights = np.vstack([1-d1/width, 1-d2/width]).T
        weights[edge,:]=0.5

        return inds, weights


    def nearest_index(self, array, value):
        return (np.abs(array-value)).argmin(axis = -1)

class SpecLibrary(ModelLibrary):
    """Class to operate on spectral libraries. Methods are provided to interpolate the
    available model spectra (stored as a structured array and a spectral array) to a certain
    set of parameters.  Subclasses are used to return the actual model spectrum given
    a set of model parameters.  Primary attributes are pars, spectra, wavelength.
    """
    
    flux_unit = 'erg/s/cm^2/AA of 1solar mass at 10pc' 
        
    def __init__(self):
        pass

    def generateSEDs(self, pars, filterlist, wave_min = 90, wave_max = 1e7,
                     keepspec = False, intspec = False, attenuator = None):
        """output is of shape (nobj,nfilter) and (nobj) and (nobj, nwave)"""
        
	maxmod = 1e7/self.wavelength.shape[0] #don't use too much memory at once
        ngrid = pars.shape[0]

	sed = np.zeros([ngrid,len(filterlist)],dtype=float)
	lbol = np.zeros(ngrid,dtype=float)
        if keepspec:
            outspectra = np.zeros([ngrid,self.wavelength.shape[0]])
        elif intspec:
            outspectra = np.zeros(self.wavelength.shape[0])
        else:
            outspectra = np.array(-1)
            
	#split big model grids to avoid memory constraints
	i=0
        while (i*maxmod <= ngrid):
            print(i)
	    s1, s2 = (i)*maxmod, np.min([(i+1)*maxmod-1,ngrid])
	    spec = self.spectra_from_pars(pars[s1:s2])
            if attenuator is not None:
                spec  = attenuator.apply_attenuation(self.wavelength, spec, pars)
	    sed[s1:s2,:] = observate.getSED(self.wavelength,spec,filterlist)
	    lbol[s1:s2] = observate.Lbol(self.wavelength,spec,wave_min,wave_max)
	    i+=1
            if keepspec is True:
                outspectra[s1:s2,:] = spec
            elif intspec is True:
                outspectra+=spec.sum(axis = 0)
                
        return sed, lbol, outspectra


    def interpolate_to_pars(self, target_points, parnames=None, itype='dt',subinds=None ):
        """Method to obtain the model spectrum for a given set of parameter values via
        interpolation of the model grid. The interpolation weights are determined
        from barycenters of a Delaunay triangulation or nLinear interpolation.
        
        The input is an array of target model parameters, optionally a string list of the
        fields containing the corresponding library model parameters
            target_points - ndarray of shape (ntarg,ndim) of ndim desired model parameters.
                            Can also be a structured array with fields named for the model parameters
            subinds - ndarray of indices of the model pars structure to use in interpolation.
                      allows for only portions of the library to be used
            parnames  - string list of the names of the model library parameters
        The output has shape (ntarg, nwave)
        """

        #Note: should (but doesn't yet) allow for grid (and point) to be scaled in any or
        #all dimensions.  is the DT invariant under scaling?  maybe the barycentric coords of simplex are
        #Note: should (but doesn't yet) check that point is in the grid


        inds, weights = self.model_weights(target_points, parnames=parnames, itype='dt',subinds = subinds)
        if subinds is not None:
            inds = subinds[inds]

        return self.combine_weighted_spectra(inds, weights)

    def combine_weighted_spectra(self, inds, weights):
        """weight self.spectra using broadcasting, then sum the weighted spectra and
        return (nwave,ntarg).  inds has shape () and weights has shape ()"""
        return (( weights* (self.spectra[inds].transpose(2,0,1)) ).sum(axis=2)).T


    def read_model_from_fitsbinary(self, filename,parnames,wavename = 'WAVE',fluxname = 'F_LAMBDA'):
        #if os.ispath(filename) is False: raise IOError('read_model_from_fitsbinary: ',filename,' does not exist')
        fits = pyfits.open( filename )
        #parse the FITS recarray and assign ModelGrid parameter, spectra, and wavelength attributes
        wavelength = fits[1].data[0][wavename]
        spectra=fits[1].data[fluxname]  #(nmod,nwave)
        #need to loop over pars and restruct to get the desired order.  Really?  Not really.  order is unimportant
        pars, partype = [], []
        for pname in parnames:
            pars.append(np.squeeze(fits[1].data[pname]))
            partype.append(fits[1].data[pname].dtype) #fix
        pars=self.structure_array(np.array(pars).transpose(),parnames,types=partype) #list ->(nmod, npar) -> Structured array

        fits.close()
        return wavelength, spectra, pars



############ A method for determining nLinear interpolation weights.  unfinished ########

#def weightsLinear(self,model_points,point):
#    """ The interpolation weights are determined from the distances to the nearest
#    grid points in each dimension.  There will be 2**ndim indices and weight products,
#    corresponding to the vertices of the (hyper)-square.  Therefore, this routine gets
#    nasty in high-dimensional spaces.  stay out of them. Requires rectilinearly gridded models.
#    see also scipy.ndimage.interpolate.map_coordinates """
#    #the n-Linear interpolation as defined here *is* invariant under rescaling of any dimension
#            
#    ndim=point.shape[0]
#        #vectorize?  allow multiple points and/or remove loop over dimensions.  need to write
#        #down the math that will speed this up. or turn into a ufunc
#    for idim in xrange(ndim):
#            #unique sorted model grid point values and
#            #distance from point (n_uniqueval)
#        model_vals = np.unique(model_thisdim)
#        dp = point[idim] - model_vals
#
#            #get the weight of the lower bounding model grid point.  The weight
#            #will be zero if grid point is higher than target (dp < 0) or grid point 
#            #one grid step or more less than target (dp/dbin >= 1) 
#        dbin = np.append(model_vals[1:]-model_vals[:-1],-1) #distance from grid point to next highest
#        w1 = (1-dp/dbin) * (dp >= 0) *  (dp/dbin < 1)
#            #if w1<0 that means target was above the whole grid, use nearest grid (which has w1<0)
#        w1 = np.where(w1 >= 0, w1,1) 
#
#            #get the weight of the upper bounding model grid point by
#            #reversing signs of above  The weight
#            #will be zero if grid point is lower than target (dp < 0) or grid point 
#            #one grid step or more more than target (dp/dbin >= 1)
#        dp = dp*(-1)
#        dbin = np.append(-1,model_vals[1:]-model_vals[:-1]) #distance from grid point to next lowest
#        w2 = (1-dp/dbin) * (dp > 0) *  (dp/dbin < 1)
#            #if w2<0 that means target was lower than the whole grid, use nearest grid (which has w1<0)
#        w2 = np.where(w2 >= 0, w2,1) 

#            #index into the model grid point values (nmod)
#        model_index = np.digitize(np.squeeze(model_points[idim,:]), model_vals)
        
#    bins=np.unique(model_points[idim,:])
#    ind=np.digitize(point[idim],bins)

#IDW weighting below:
#dists = np.sqrt( ( (dtri.points[inds]-point)**2 ).sum(1) )
#weights = 1.0/dists
#interpolate only if you need to.  should probably check before interpolation is run
#(or triangulation) but this is a failsafe
#exact = np.where(dists == 0.)
#if (exact[0].shape[0] > 0):
#    weights = np.array([1.0])
#    inds = inds[exact[0]]
#need to renormalize

#def testWeighting():
#    m=np.array([0.,1,2,3])
#    t=1.4
#    #assert (self.weightsLinear(m,t) is [1,2] ,[0.6,0.4])
#    t=1
#    #assert (self.weightsLinear(m,t) is [1] ,[1])
#    m=np.array([0,1,1,2,3])
#        
#    raise ValueError('No test written')
#    pass
