from pyscf import gto
from pyscf import dft
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~methods (and  such) begin here
class MOData:
    '''
    
    Object storing all the useful values needed for the other parts. Must be instantiated using the build_modat() method below
    
    '''
    def __init__(self, energies, occupancies, coefficients, ao_labels):
        self.energies = energies
        self.occupancies = occupancies
        self.coefficients = coefficients
        self.ao_labels = ao_labels

def build_modat(mf):
    """
    Returns the MOData object from the mean field (mf) of the mol object 

    Parameters
    ----------
    mf : dft.RKS or dft.UKS object depending on if it is closed shell or open shell, repsectively
        The main object the calcs are applied to. ***NOT*** where to find the calcs, use the returned MOData object for easy access to the data.

    Returns
    -------
    MOData
        useful box of all the info needed (hopefully)

    Raises
    ------
    N/A
    """
    energy = mf.mo_energy
    coeffs = mf.mo_coeff
    if mf.mo_coeff.ndim == 1:
        coeffs = mf.mo_coeff
    else: 
        coeffs = mf.mo_coeff[0]
    if mf.mo_energy.ndim == 1:
        energy = mf.mo_energy
    else:
        energy = mf.mo_energy[0]
    return  MOData(energy*27.211, mf.mo_occ, coeffs, mf.mol.ao_labels())