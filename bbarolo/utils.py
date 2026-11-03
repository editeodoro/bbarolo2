"""
This module defines some utility functions and classes  
"""

########################################################################
# Copyright (C) 2021 Enrico Di Teodoro
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
########################################################################

import os
import numpy as np
from astropy.io import fits


def isIterable(p):
    """Check if p is an iteratable (list,tuple or numpy array). """
    return isinstance(p, (list, tuple, np.ndarray))


def isNumber(p):
    """Check if p is a number. """
    return isinstance(p, (float, int, np.integer, np.floating))


def emptyfits(axisDim,cdelts,crpixs=None,crvals=None,ctypes=None,cunits=None,\
              data=None,beam=None,obj=None,bunit=None,dtype=np.float32):
    
    """ Defines an empty fits file.
    
        Args:
          axisDim (list):    axis sizes
          cdelts (list):     pixel sizes
          crpixs (list):     central pixels (default: axisDim/2.+1)
          crvals (list):     central values (default: 0)
          ctypes (list):     axis types (default: ra, dec, velo-helo)
          cunits (list):     axis units (default: deg, deg, km/s)
          beam (float,list): beam size in degrees
          obj (str):         name of the object
          bunit (str):       flux unit
          dtype (type):      type of data (default: float32)
    
        Returns:
          A fits.PrimaryHDU() object 
    """
    # Default cunits and ctypes
    ctypes = ['RA---SIN', 'DEC--SIN', 'VELO-HEL'] if ctypes is None else ctypes
    cunits = ['deg', 'deg', 'km/s'] if cunits is None else cunits
    crvals = [60, 0, 0] if crvals is None else crvals
    
    axisDim = np.array(axisDim)
    
    # Define new header
    h = fits.Header()
    # Adding WCS informations
    for i in range (len(axisDim)):
        h['CRPIX%d'%(i+1)] = (axisDim[i]-1)/2.+1 if crpixs is None else crpixs[i]
        h['CRVAL%d'%(i+1)] = crvals[i]
        h['CDELT%d'%(i+1)] = cdelts[i]
        h['CUNIT%d'%(i+1)] = cunits[i]
        h['CTYPE%d'%(i+1)] = ctypes[i]
    
    if h['CDELT1']>0:
        print ("WARNING, CDELT1 should be negative!")
        
    # Adding beam information
    if beam is not None:
        bmaj = bmin = bpa = 0
        if isNumber(beam): 
            bmaj, bmin, bpa = beam, beam, 0
        elif isIterable(beam) and len(beam)<4:
            if   len(beam)==3: bmaj, bmin, bpa = beam
            elif len(beam)==2: bmaj, bmin, bpa = beam[0], beam[1], 0
            elif len(beam)==1: bmaj, bmin, bpa = beam[0], 0, 0
        h['BMAJ'],h['BMIN'],h['BPA'] = bmaj, bmin, bpa
    
    h['OBJECT'] = 'NONE'
    if obj is not None: h['OBJECT'] = obj
    if bunit is not None: h['BUNIT'] = bunit
    h['EQUINOX'] = 2000.
    
    # Defining empty data
    if data is None:
        d = np.zeros(shape=np.flip(axisDim),dtype=dtype)
    else:
        ok = True
        for i in range (len(axisDim)):
            ok *= np.flip(axisDim)[i]==data.shape[i]
        if ok:  d = data
        else: raise ValueError("Array dimensions are different than axisDim")
    
    return fits.PrimaryHDU(data=d.astype(dtype),header=h)
    

class SimulatedGalaxyCube(object):
    """ 
    A class to easily simulate an emission line datacube for a  
    galaxy with some parameters
    
    Attributes
    ----------
    f: (astropy.PrimaryHDU)
      A FITS HDU representing the emission line datacube 
    
    radii: (list, np.array)
      Radii of the galaxy model
    
    dens: (list, np.array)
      Surface-density (or flux) profile of the galaxy
    
    inc: (list, np.array)
      Inclination angles of the galaxy
    
    pa: (list, np.array)
      Position angles of the galaxy
    
    z0: (list, np.array)
      Scale heights of the galaxy
    
    vrot: (list, np.array)
      Rotation curve of the galaxy
    
    disp: (list, np.array)
      Velocity dispersions of the galaxy
    
    vrad: (list, np.array)
      Radial velocities of the galaxy

    xpos: (list, np.array)
      X center of the galaxy
    
    ypos: (list, np.array)
      Y center of the galaxy
    
    vsys: (list, np.array)
      Systemic velocity of the galaxy
    
    model: (astropy.PrimaryHDU)
      The model datacube, set by run() (with noise, if requested). The header is
      that of the template cube.
    
    model_nonoise: (astropy.PrimaryHDU)
      The model datacube without noise, set by run()
    
    noiserms: (float)
      RMS of the noise added to the model by run()
    
    
    Methods
    -------
    define_galaxy(...):
      Define the parameters of the galaxy model
    
    savefile(fileout):
      Write galaxy parameters to a text file
    
    run(noise,smooth,threads,seed,verbose,stdout,**kwargs):
      Build the emission-line datacube model with GALMOD (in memory): sets model.
    
    write_model(fileout,overwrite,nonoise,params):
      Write the model datacube to a FITS file.
    
    """
    def __init__(self,fitscube=None,axisDim=None,cdelts=None,**kwargs):
        """ Initialize a SimulatedGalaxyCube instance.
            Here we define the template cube that will be used to create
            the galaxy model.
            If no inputs are given, a default template cube will be used.
            
        Args:
          fitscube (str or astropy.fits): template cube. If None, a default cube will 
                                          be created from axisDim and cdelts.
          axisDim (list):   axis dimensions of the template cube
          cdelts (list):    pixel dimensions of the template cube
          kwargs (dict):    Additional parameters to be passed to emptyfits()

        """
        if fitscube is not None:
            if isinstance(fitscube, str):
                self.f = fits.open(fitscube)[0]
            elif isinstance(fitscube, fits.PrimaryHDU):
                self.f = fitscube
            else:
                raise ValueError("fitscube must be a string or an astropy.io.fits.PrimaryHDU")
        else:
            # A default cube size if not given
            if not axisDim: axisDim = [256,256,64]
            if not cdelts: cdelts = [-20./3600.,20./3600.,10]
            self.f     = emptyfits(axisDim,cdelts,**kwargs)
        
        # The model (set by run())
        self.model = None
        self.model_nonoise = None
        self.noiserms = 0.
        
        # Ring parameters
        self.radii = None
        self.dens  = None
        self.inc   = None
        self.pa    = None
        self.z0    = None
        self.vrot  = None
        self.disp  = None
        self.vrad  = None
        self.xpos  = None
        self.ypos  = None
        self.vsys  = None
        
    
    
    def define_galaxy(self,radmin=None,radmax=None,radii=None,dens=None,\
                      ishole=None,xpos=None,ypos=None,vsys=None,\
                      inc=None,warpinc=None,mininc=30,maxinc=89,maxiwarp=15,\
                      pa=None,warppa=None,minpa=0,maxpa=359,maxpawarp=30,\
                      vrot=None,vrotdwarf=None,vdisp=None,z0=None,flare=False,\
                      vrad=None,radmotion=False,maxvrad=20):
        """ Define the parameters of the model galaxy and sets all the variables
            needed by the class. This function needs to be called before run().
                      
            All inputs are optional. The user can decide to set any of them to 
            control the parameters of the galaxy model. If none is set, the galaxy
            will be completely random. 
            
        Args:
          radmin (float):      Minimum radius of the galaxy (arcsec).
          radmax (float):      Maximum radius of the galaxy (arcsec).
          radii (list):        List of radii (arcsec). If None, assume cubesize/3.
          dens (float,list):   Surface density (1E20 cm-2). If None, assume a profile.
          ishole (bool):       Whether to have a central hole in dens. If None, randomly decide.
          xpos (float,list):   X center (pixel). If None, assume the center of Xsize/2.
          ypos (float,list):   Y center (pixel). If None, assume the center of Ysize/2.
          vsys (float,list):   Systemic velocity (pixel). If None, assume the Zsize/2.
          inc (float,list):    Inclination angle (deg). If None, assume a random inc.
          warpinc (bool):      Whether to set a warp in inclination. If None, randomly decide.
          mininc (float):      Minimum inclination for random draw when inc=None.
          maxinc (float):      Maximum inclination for random draw when inc=None.
          maxiwarp (float):    Maximum deviation in the inc warp if warpinc=True.
          pa (float,list):     Position angle (deg). If None, assume a random PA.
          warppa (bool):       Whether to set a warp in PA. If None, randomly decide.
          minpa (float):       Minimum PA for random draw when pa=None.
          maxpa (float):       Maximum PA for random draw when pa=None.
          maxpawarp (float):   Maximum deviation in the PA warp if warppa=True.
          vrot (float,list):   Rotation velocity (km/s). If None, assume a random curve.
          vrotdwarf (bool):    Whether to assume a dwarf-like rotation curve if vrot=None.
          vdisp (float,list):  Velocity dispersion (km/s). If None, assume a random vdisp.
          z0 (float,list):     Disk scale height (arcs). If None, assume a random z0.
          flare (float,list):  Whether to set a flare in z0. If None, randomly decide.
          vrad (float,list):   Radial velocity (km/s). If None, assume 0.
          radmotion (float,list): Whether to set some radial motion. If None, randomly decide.
          maxvrad (float):     Maximum deviation in the vrad if radmotion=True.

        """
        
        h = self.f.header
        pixsize = np.mean([np.abs(h['CDELT1']),np.abs(h['CDELT2'])])*3600.
        
        if radii is None and radmax is None:
            # Setting a maximum radius at cube dimension/3
            radmax = np.round(h['NAXIS2']*pixsize/3.)

        # Defining radii
        if isIterable(radii): self.radii = radii
        else:
            radmin = radmin if radmin else pixsize
            self.radii = np.arange(radmin,radmax+pixsize,pixsize)
        
        radii = self.radii
        radmax = radii[-1]
        
        
        # Defining X centre 
        if isIterable(xpos):
            if len(xpos)==len(radii): self.xpos = xpos
            else: 
                raise AttributeError("radii and xpos must have the same size") 
        elif isNumber(xpos):
            self.xpos = np.full(len(radii),xpos)
        else:
            # Centering the galaxy
            self.xpos = np.full(len(radii),h['NAXIS1']/2.)
            
        
        # Defining Y centre 
        if isIterable(ypos):
            if len(ypos)==len(radii): self.ypos = ypos
            else: 
                raise AttributeError("radii and ypos must have the same size") 
        elif isNumber(xpos):
            self.ypos = np.full(len(radii),ypos)
        else:
            # Centering the galaxy
            self.ypos = np.full(len(radii),h['NAXIS2']/2.)
            
        
        # Defining systemic velocity 
        if isIterable(vsys):
            if len(vsys)==len(radii): self.vsys = vsys
            else: 
                raise AttributeError("radii and ypos must have the same size") 
        elif isNumber(vsys):
            self.vsys = np.full(len(radii),vsys)
        else:
            # Centering the galaxy in velocity
            vs = (h['NAXIS3']/2.+1-h['CRPIX3'])*h['CDELT3']+h['CRVAL3']
            self.vsys = np.full(len(radii),vs)
        
            
        # Defining density profile
        if isIterable(dens):
            if len(dens)==len(radii): self.dens = dens
            else: 
                raise AttributeError("radii and dens must have the same size") 
        elif isNumber(dens):
            self.dens = np.full(len(radii),dens)
        else:
            # Using a random densifity profile
            d0 = 50.
            R0 = radmax/5.
            if ishole is None: ishole =  np.random.choice([True, False])
            if ishole:
                self.dens = d0*np.exp(-radii/R0
                            -R0/2./(0.5*radii+R0/4.))
            else: self.dens = d0*np.exp(-radii/R0)
        
        
        # Defining inclination
        if isIterable(inc):
            if len(inc)==len(radii): self.inc = inc
            else: 
                raise AttributeError("radii and inc must have the same size")
        else:
            if isNumber(inc):
                self.inc = np.full(len(radii),inc)
            else:
                # Using a random inclination
                i = np.random.uniform(mininc,maxinc)
                self.inc = np.full(len(radii),np.round(i))
            
            # Creating a warp in inclination if requested
            if warpinc is None: warpinc = np.random.choice([True, False])
            if warpinc:
                rstart  = np.random.randint(0,len(radii)-1)
                start = self.inc[0]
                stop  = np.random.uniform(start-maxiwarp,start+maxiwarp)
                mline = (stop-start)/(radmax-radii[rstart])
                self.inc[radii>radii[rstart]] = mline*(radii[radii>radii[rstart]]-radmax)+stop
        
        
        # Defining position angle
        if isIterable(pa):
            if len(pa)==len(radii): self.pa = pa
            else: 
                raise AttributeError("radii and pa must have the same size")
        else:
            if isNumber(pa):
                self.pa = np.full(len(radii),pa)
            else:
                # Using a random position angle
                p = np.random.uniform(minpa,maxpa)
                self.pa = np.full(len(radii),np.round(p))
        
            # Creating a warp in position angle if requested
            if warppa is None: warppa = np.random.choice([True, False])
            if warppa:
                rstart  = np.random.randint(0,len(radii)-1)
                start = self.pa[0]
                stop  = np.random.uniform(start-maxpawarp,start+maxpawarp)
                mline = (stop-start)/(radmax-radii[rstart])
                self.pa[radii>radii[rstart]] = mline*(radii[radii>radii[rstart]]-radmax)+stop
                
        
        # Defining scale height
        if isIterable(z0):
            if len(z0)==len(radii): self.z0 = z0
            else: 
                raise AttributeError("radii and z0 must have the same size")
        else:
            if isNumber(z0):
                self.z0 = np.full(len(radii),z0)
            else:
                # Random scale height
                z = np.random.uniform(0,3*pixsize)
                self.z0 = np.full(len(radii),z)
            
            # Creating a flare if requested
            if flare is None: flare = np.random.choice([True, False])
            if flare:
                rstart  = np.random.randint(0,len(radii)-1)
                start = self.z0[0]
                stop  = np.random.uniform(start,start+start)
                mline = (stop-start)/(radmax-radii[rstart])
                self.z0[radii>radii[rstart]] = mline*(radii[radii>radii[rstart]]-radmax)+stop
            

        # Defining rotation curve 
        if isIterable(vrot):
            if len(vrot)==len(radii): self.vrot = vrot
            else: 
                raise AttributeError("radii and vrot must have the same size")
        elif isNumber(vrot):
            self.vrot = np.full(len(radii),vrot)
        else:
            # Random rotation curve
            if vrotdwarf is None: vrotdwarf = np.random.choice([True, False])
            if vrotdwarf:
                vflat = np.random.uniform(10,130)
                rs = np.random.uniform(pixsize,radmax/2)
            else:
                vflat = np.random.uniform(100,350)
                rs = np.random.uniform(pixsize,radmax/10)
            self.vrot  = 2./np.pi*vflat*np.arctan(radii/rs)
            self.rs = rs
            self.vflat = vflat 
            
            
        # Defining velocity dispersion 
        if isIterable(vdisp):
            if len(vdisp)==len(radii): self.disp = vdisp
            else: 
                raise AttributeError("radii and vdisp must have the same size")
        elif isNumber(vdisp):
            self.disp = np.full(len(radii),np.round(vdisp))
        else:
            # Random constant dispersion
            vd = np.random.uniform(6,15)
            self.disp = np.full(len(radii),np.round(vd)) 
        
        
        # Defining radial motions
        if isIterable(vrad):
            if len(vrad)==len(radii): self.vrad = vrad
            else: 
                raise AttributeError("radii and vrad must have the same size")
        else:
            if isNumber(vrad):
                self.vrad = np.full(len(radii),vrad)
            else:
                # No radial motions 
                self.vrad = np.full(len(radii),0.)
            
            if radmotion is None: radmotion = np.random.choice([True, False])
            if radmotion:
                rstart  = np.random.randint(0,len(radii)-1)
                start = self.vrad[0]
                stop  = np.random.uniform(start-maxvrad,start+maxvrad)
                mline = (stop-start)/(radmax-radii[rstart])
                self.vrad[radii>radii[rstart]] = mline*(radii[radii>radii[rstart]]-radmax)+stop
            
            

    def savefile(self,fileout="galaxy_params.txt"):
        """ Save galaxy parameter to a text file.
            
        Args:
          fileout (str):  Name of text file
        """
        if self.radii is not None:
            np.savetxt(fileout,np.transpose([self.radii,self.vrot,self.vrad,self.disp,self.inc,
            self.pa,self.z0,self.dens,self.xpos,self.ypos,self.vsys]),
            header="RADII         VROT    VRAD   VDISP      INC       PA      Z0      DENS    XPOS    YPOS    VSYS",
            fmt='%-12.5f %7.1f %7.1f %7.1f %8.2f %8.2f %7.2f %9.2e %7.1f %7.1f %7.1f')
        
        
        
    def run(self,noise=None,smooth=True,threads=None,seed=None,verbose=False,stdout=None,**kwargs):
        """ Build the model datacube of the galaxy defined with define_galaxy().
        
            The model is the one of the GALMOD task with NORM=NONE (bbarolo.GalMod.model(),
            in memory) and is stored in self.model, an astropy PrimaryHDU with the header of
            the template cube (self.model_nonoise: the same without noise). Nothing is
            written to disk: use write_model() for that.
            
        Args:
          noise (float, bool): RMS noise of the output datacube (in the units of the cube).
                               If True, a random value between 0 and 0.05.
          smooth (bool):       Whether to smooth the model to the resolution of the template cube
          threads (int):       Number of threads (default: THREADS, the number of cores, at most 16)
          seed (int):          Seed for the noise (default: random)
          verbose (bool):      Whether to print BBarolo's messages
          stdout (str):        Where to print BBarolo's messages: None (screen), a file name or 'null'
          kwargs (dict):       Other GALMOD parameters (e.g. ltype, cdens, nv, iseed, galmodtype,
                               ringmode), with the keywords of a BBarolo parameter file.
        
        Returns:
          The model (astropy PrimaryHDU), also in self.model
        """
        if self.radii is None:
            raise RuntimeError("Please, define a model with define_galaxy() before calling run()")
        if 'exe' in kwargs:
            raise TypeError("SimulatedGalaxyCube.run() builds the model in Python: it no longer "
                            "runs a BBarolo executable (remove the 'exe' argument).")
        if 'outfolder' in [k.lower() for k in kwargs]:
            raise TypeError("SimulatedGalaxyCube.run() does not write files any more: "
                            "use write_model(fileout) after run().")
        
        # Noise: True = a random rms; a number = that rms; None/False = no noise
        # (True is checked first: a bool is also an int)
        noiserms = 0.
        if isinstance(noise, (bool, np.bool_)):
            if noise: noiserms = np.random.uniform(0,0.05)
        elif isNumber(noise): noiserms = float(noise)
            
        try: self.f.header['BMAJ']>0
        except: smooth = False
        
        from .tasks import GalMod, _canonical, _stdout_to
        
        # GALMOD parameters (kwargs override them; keywords are case-insensitive)
        opts = dict(radii=self.radii, vrot=self.vrot, vrad=self.vrad, vdisp=self.disp,
                    inc=self.inc, pa=self.pa, z0=self.z0, dens=self.dens, xpos=self.xpos, ypos=self.ypos,
                    vsys=self.vsys, norm='NONE', mask='NONE', empty=False, linear=0, cdens=10,
                    sm=smooth, verbose=verbose)
        if threads is not None:
            opts['threads'] = threads
        prm = _canonical({k: np.asarray(v, dtype=float) if isIterable(v) else v for k, v in opts.items()})
        prm.update(_canonical(kwargs))
        h = self.f.header
        if 'BMAJ' not in h and not any(k in prm for k in ('BMAJ', 'BEAMFWHM')):
            raise ValueError("The template cube has no beam: give it when creating the SimulatedGalaxyCube "
                             "(beam=...), or pass bmaj=... (arcsec) to run().")
        
        with _stdout_to(stdout if stdout is not None or verbose else 'null'):
            gm = GalMod(self.f, **prm)
            clean = gm.model()
            noisy = gm.diskmodel.addnoise(clean.data, noiserms, rng=np.random.default_rng(seed))
        
        self.noiserms = noiserms
        self.model_nonoise = fits.PrimaryHDU(data=np.asarray(clean.data, dtype=np.float32), header=h.copy())
        self.model = fits.PrimaryHDU(data=noisy, header=h.copy())
        return self.model
    
    
    def write_model(self,fileout=None,overwrite=True,nonoise=False,params=False):
        """ Write the model datacube (built by run()) to a FITS file.
            
        Args:
          fileout (str):    Name of the FITS file. Default: <OBJECT>mod.fits
          overwrite (bool): Whether to overwrite an existing file
          nonoise (bool):   Write the model without noise (model_nonoise)
          params (bool):    Also write the galaxy parameters (see savefile()) in
                            <fileout without .fits>_params.txt
        
        Returns:
          The name of the FITS file
        """
        if self.model is None:
            raise RuntimeError("Please, build the model with run() before calling write_model()")
        if fileout is None:
            obj = str(self.f.header.get('OBJECT', 'galaxy')).strip() or 'galaxy'
            fileout = f"{obj}mod.fits"
        d = os.path.dirname(fileout)
        if d: os.makedirs(d, exist_ok=True)
        hdu = self.model_nonoise if nonoise else self.model
        hdu.writeto(fileout, overwrite=overwrite)
        if params:
            base = fileout[:-5] if fileout.lower().endswith('.fits') else fileout
            self.savefile(fileout=f"{base}_params.txt")
        return fileout
