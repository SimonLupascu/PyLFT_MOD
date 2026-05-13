from pyscf import gto
from pyscf import dft
from rdkit import Chem
from rdkit.Chem import AllChem
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from PyLFT_MOD.src.pylft_mod.distortions import classify_distortion
from pyscf_tools import get_d_count, get_ligand_charge, find_spin

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~methods begin here
def build_complex(atoms, metal, ligands, ox_state, basis = 'def2-svp'):
    """
    Returns the mol object upon which further calculations can be performed

    Parameters
    ----------
    atoms : list (I think...?)
        The whole thing sent from architector with coords and all, (e.g. [symbol, x, y, z] entries ------> [['Fe', 0, 0, 0], ['N', 2, 0, 0], ...])
    metal : str
        Atomic symbol of the metal center (e.g. 'Fe', 'Co', etc.)
    ligands : list
        All the ligands coordinated to the metal. Repeats must be mentioned (e.g. ['NH3', 'NH3', 'NH3', 'OH-', 'OH-', 'I-'])
    ox_state : int
        The oxidation state of the metal center
    basis : str
        The basis set for the orbitals (e.g. 'sto-3g' for slater type orbital app. with three gaussians) Automatically set to 'def2-svp' for complex calcs due to complexity of d orbitals

    Returns
    -------
    mol
        mol object which the mean field of will be calculated later with get_mf

    Raises
    ------
    N/A
    """
    mol = gto.Mole()
    mol.atom = atoms
    if classify_distortion(metal, ox_state, ligands)['target_group'] != None:
        mol.symmetry = False
    else:
        mol.symmetry = True
    mol.charge = ox_state + sum(get_ligand_charge(lig) for lig in ligands)
    mol.spin = find_spin(metal, get_d_count(metal, ox_state), ligands)
    mol.basis = basis
    mol.build()
    return mol

def get_mf(mol):
    """
    Returns the mean field app. based on the RKS or UKS depending on open or closed shell system

    Parameters
    ----------
    mol : mol
        the mol object obtained from the build_complex() method

    Returns
    -------
    rks_mf or uks_mf depending on closed or open shell
        mf app. which will be acted upon further in the modata.py file methods and MOData dataclass

    Raises
    ------
    N/A
    """
    func = 'b3lyp'
    if mol.spin == 0:
        rks_mf = dft.RKS(mol)
        rks_mf.xc = func
        rks_mf.kernel()
        return rks_mf
    else:
        uks_mf = dft.UKS(mol)
        uks_mf.xc = func
        uks_mf.kernel()
        return uks_mf