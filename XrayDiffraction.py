"""
Code example for the simulation of electron diffraction patterns based on molecular geometries 
within the independent atom model. The simulations yield results compatible with the electron 
beam parameters of the MeV Ultrafast Electron Diffraction (UED) facility at SLAC National 
Accelerator Laboratory (https://lcls.slac.stanford.edu/instruments/mev-ued). 
Created by Thomas Wolf, 02/26/2020
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

############################################################################################################
## Classes and functions ###################################################################################
############################################################################################################

class mol_geom():
    """
    Creates a molecular geometry object.
    Arguments: 
    filename: Path to a molecular geometry (*.xyz) file. See below for the file format expected
    by the code.
    """
    def __init__(self,filename):
        """
        Function to initialize the geometry object by loading geometry data.
        """
        self.loadxyz(filename)
        
    def loadxyz(self,filename):
        """
        Function to load geometry data from an *.xyz file. The code assumes the file synthax 
        as read and written by programs like Molden (http://cheminf.cmbi.ru.nl/molden/). It 
        ignores the first two lines of the *.xyz file. The first line usually contains the 
        number of atoms in the molecular geometry, the second line contains comments. The code
        expects a line for each atom of the molecular geometry in the remainder of the file.
        Each line contains the following information in the exact order: Element letter, x, y, and z 
        coordinates. The different items are separated by spaces.
        """
        # Load geometry file as strings
        with open(filename,'r') as geofile:
            geostr = geofile.readlines()

        # Extract element information (elements) and coordinates (geom)
        geostr2 = geostr[2:]
        self.coordinates = np.zeros((len(geostr)-2,3))
        self.elements = []
        for i in np.arange(len(geostr2)):
            arr = geostr2[i].split()
            self.elements.append(arr[0])
            self.coordinates[i,0] = float(arr[1])
            self.coordinates[i,1] = float(arr[2])
            self.coordinates[i,2] = float(arr[3])
            
############################################################################################################
            
class Atomic_Scattering_Cross_Sections():
    """
    Creates an object containing form factors for different elements. This class currently 
    supports the following elements: H, He, C, N, O, F, S, Fe, Br, I. The form factors are
    calculated with the ELSEPA program (https://github.com/eScatter/elsepa) assuming the 
    standard electron kinetic energy of 3.7 MeV used at the SLAC UED facility. This class must
    be modified to add unsupported elements.
    """
    def __init__(self):
        """
        Function to initialize the form factor object by loading form factors.
        """
        # This line must be edited to add elements:
        self.supported_elements = ['H', 'He', 'C', 'N', 'O', 'F', 'S', 'Fe', 'Br', 'I']
        for element in self.supported_elements:
            exec('self.' + element +", self.thetadeg = self.load_form_fact('" + element + "')")
    
    def load_form_fact(self,Element):
        """
        Function to load the scattering form factor for a specific element from an ELSEPA
        output file. 
        Arguments:
        Element:  Element symbol as string
        Returns:
        FF:       Angle-dependent scattering intensity in units of a0^2/sr
        thetadeg: Scattering angle in degrees
        """
        if len(Element)<2:
            Element = Element + ' '
            
        with open(Element + '3p7MeV.dat') as f:
            lines = f.readlines()
        
        for i,line in enumerate(lines):
            if line.find('#')!=-1:
                continue
            else:
                break
                
        lines = lines[i:]
        thetadeg = np.zeros((len(lines),))
        FF = np.zeros_like(thetadeg)
        for i in np.arange(len(lines)):
            thetadeg[i] = (float(lines[i].split()[0]))
            FF[i] = (float(lines[i].split()[3]))
        return FF, thetadeg
        
from scipy.interpolate import interp1d

############################################################################################################

class Diffraction():
    """
    Creates a diffraction object.
    Arguments:
    geom:   mol_geom object
    AtScatXSect: Scattering cross-section object
    Npixel: Length of Q-array
    Max_Q:  Maximum Q in inverse Angstroms
    """
    def __init__(self,geom,AtScatXSect,Npixel=120,Max_s=12):
        """
        Function to initialize Diffraction object.
        """
        self.coordinates = geom.coordinates
        self.elements = geom.elements
        self.AtScatXSect = AtScatXSect
        self.U = 3.7 # Electron kinetic energy
        self.Max_s = Max_s
        self.Npixel = Npixel
        
        E=self.U*1e6*1.6022*1e-19
        m=9.1094e-31
        h=6.6261e-34
        c=299792458

        lambdaEl=h/np.sqrt(2*m*E)/np.sqrt(1+E/(2*m*c**2)) # Electron wavelength
        k=2*np.pi/lambdaEl # Electron wave vector

        thetarad = self.AtScatXSect.thetadeg/360*2*np.pi
        self.a = 4*np.pi/lambdaEl*np.sin(thetarad/2)/1E10
        
        
    def make_1D_diffraction(self):
        """
        Function to create a 1D diffraction pattern assuming an ensemble of randomly oriented
        molecules.
        """
        natom = len(self.elements)
        self.s = np.linspace(0,np.float(self.Max_s),self.Npixel)

        

        self.I_at_1D = np.zeros((len(self.s),)) # Atomic scattering contribution to diffraction signal
        fmap = []
        for element in self.elements:
            namespace = {'interp1d'}
            f = eval('interp1d(self.a,np.sqrt(self.AtScatXSect.' + element + '))')
            fmap.append(f(self.s))
            self.I_at_1D += np.square(abs(f(self.s)))

        # Contribution from interference between atoms to diffaction signal:
        self.I_mol_1D = np.zeros_like(self.I_at_1D) 
        for i in np.arange(natom):
            for j in np.arange(natom):
                if i!=j:
                    dist = np.sqrt(np.square(self.coordinates[i,:]-self.coordinates[j,:]).sum())
                    self.I_mol_1D += abs(fmap[i])*abs(fmap[j])*np.sin(dist*self.s)/(dist*self.s)

        self.sM_1D = self.s*self.I_mol_1D/self.I_at_1D # Modified molecular diffraction
        
    def make_2D_diffraction(self):
        """
        Function to create a 2D diffraction pattern assuming an ensemble of randomly oriented
        molecules.
        """
        self.sy,self.sz = np.meshgrid(np.arange(-1*self.Max_s,self.Max_s,2*self.Max_s/self.Npixel), \
                            np.arange(-1*self.Max_s,self.Max_s,2*self.Max_s/self.Npixel))
        self.sr = np.sqrt(np.square(self.sy)+np.square(self.sz))
        natom = len(self.elements)

        self.I_at_2D = np.zeros_like(self.sr) # Atomic scattering contribution to diffraction signal
        fmap = []
        for element in self.elements:
            f = eval('interp1d(self.a,np.sqrt(self.AtScatXSect.' + element + '))')
            fmap.append(f(self.sr))
            self.I_at_2D += np.square(abs(f(self.sr)))

        # Contribution from interference between atoms to diffaction signal:
        self.I_mol_2D = np.zeros_like(self.I_at_2D) 
        for i in np.arange(natom):
            for j in np.arange(natom):
                if i!=j:
                    dist = np.sqrt(np.square(self.coordinates[i,:]-self.coordinates[j,:]).sum())
                    self.I_mol_2D += abs(fmap[i])*abs(fmap[j])*np.sin(dist*self.sr)/(dist*self.sr) 

        self.sM_2D = self.sr*self.I_mol_2D/self.I_at_2D # Modified molecular diffraction
        

