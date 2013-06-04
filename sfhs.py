import pyfits
import numpy as np 
import scipy.special
#import bandpasses

class SFHComponent(object):
    """This class stores and operates on SF components with parameterized functional shapes, including
    top hats ('tophat'), exponentials ('exp'), delayed exponentials ('delay'), instantaneous bursts ('bursts').
    Methods are included to integrate the SFR over a given age (lookback time) range and to return the
    SFR at a given lookback time (age)."""

    def __init__(self,age=13,tau=1,norm=1, Z=1,user_sfr=0):
        self.age = age  #lookback time of the beginning of the SF
        self.tau = tau  # characteristic timescale of the SFH component
        self.norm=norm
        self.Z=Z        # metallicity of the SFH component
        self.user_sfr=user_sfr
        self.add_norm()
        if sftype == 'tophat':
            self.comp_sfr=norm # normalization of the SFH component (in SFR)
            self.comp_totmass=norm*self.tau
        else:
            self.comp_totmass = norm # normalization of the SFH component (in total stellar mass formed)

    def integrated_SFR(self,target_ages,target_ages_start=-1):
        """integratedSFR
        calculates the SFR integrated from the beginning of the SF to the target 'age',
        where 'age' defined as the lookback time in the frame of the galaxy.  Optionally,
        an age/lookback time at which to start the integration can be supplied """
        tprime=self.age-target_ages #reverse time scale to component frame, not lookback time of the galaxy
        phase=tprime/self.tau

        #if necessary, do the integration up the beginning of the age range
        #by calling the method again, i.e. recur
        mstart=np.zeros(target_ages_start.shape)
        ind_start=np.where(target_ages_start > 0 and tprime > 0)
        mstart[ind_start]=self.integrated_SFR(target_ages_start[ind_start])

        mass=np.zeros(target_ages.shape)
        #return zero if the desired age is before the component started
        ind_valid=np.where(tprime > 0) #add the function to the where clause to skip some steps?
        #integrate up to the specified lookback time
        mass[ind_valid]=self.par_mass(phase[ind_valid],mstart=mstart[ind_valid])
        return mass
        
    def SFR(self,target_ages):
        tprime = self.age-target_ages #reverse time scale to component frame, not lookback time of the galaxy
        phase = tprime/self.tau
        ind_good = np.where(tprime > 0) #add the function to the where clause to skip some steps?
        sfr = np.zeros(phase.shape)
        sfr[ind_good]=self.par_SFR(phase[ind_good])
        return sfr


class TopHat(SFHComponent):
    def par_SFR(self,phase):
        return self.comp_sfr*np.where(phase < 1,1,0)
    def par_mass(self,phase,mstart = 0):
        return self.comp_sfr*np.where(phase<1,phase,1) - mstart
    def age_array(self,ages):
        pass

class Delay(SFHComponent):
    def par_sFR(self,phase):
        return self.comp_totmass*(phase/self.tau)*np.exp(0-phase)        
    def par_mass(self,phase,mstart = 0):
        return self.comp_totmass*scipy.special.gammainc(2,phase) - mstart
    def age_array(self,ages):
        pass

class Exp(SFHComponent):
    def par_SFR(self,phase):
        return self.comp_totmass/self.tau*np.exp(0-phase)
    def par_mass(self,phase,mstart = 0):
        return self.comp_totmass*np.exp(0-phase) - mstart
    def age_array(self,ages):
        pass

class Burst(SFHComponent):
    def par_SFR(self,phase):
        return np.where(phase == 0,1,0)
    def par_mass(self,phase,mstart = 0):
        return self.comp_totmass-mstart
    def age_array(self,ages):
        pass

class UserSFH(SFHComponent):
    #This needs to be checked/fixed!!!!!
    def par_SFR(self,phase):
        return np.interp(self.user_sfr,self.age,phase)
    def par_mass(self,phase,mstart = 0):
        return np.trapz(self.user_sfr[ind_valid],self.age[ind_valid],axis=1)-mstart
    def age_array(self,ages):
        pass

