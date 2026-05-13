from pyscf import gto
from pyscf import dft
from PyLFT_MOD.src.pylft_mod.distortions import get_spin_state
#~~~~~~~~~~~~~~~~~~~ dicts required for some info
high_spins = {
        4 : 2,
        5 : 2.5,
        6 : 2,
        7 : 1.5,
        0 : 0,
        1 : 0.5,
        2 : 1,
        3 : 1.5,
        8 : 1,
        9 : 0.5,
        10 : 0
}
low_spins = {
        4: 1,
        5: 0.5,
        6: 0,
        7: 0.5,
        0 : 0,
        1 : 0.5,
        2 : 1,
        3 : 1.5,
        8 : 1,
        9 : 0.5,
        10 : 0
}
ligand_charges = {
        'I-': -1,
        'Br-': -1,
        'Cl-': -1,
        'F-': -1,
        'NH3': 0,
        'H2O': 0,
        'CN-': -1,
        'CO': 0,
        'O2-' : -2,
        'OH-' : -1
    }
d_electrons = {
    'Ti': {2: 2, 3: 1, 4: 0},
    'V':  {2: 3, 3: 2, 4: 1, 5: 0},
    'Cr': {2: 4, 3: 3, 6: 0},
    'Mn': {2: 5, 3: 4, 4: 3, 7: 0},
    'Fe': {2: 6, 3: 5},
    'Co': {2: 7, 3: 6},
    'Ni': {2: 8, 3: 7},
    'Cu': {1: 10, 2: 9, 3: 8},
    'Zn': {2: 10},
    'Zr': {2: 2, 3: 1, 4: 0},
    'Nb': {3: 2, 4: 1, 5: 0},
    'Mo': {2: 4, 3: 3, 4: 2, 5: 1, 6: 0},
    'Tc': {2: 5, 3: 4, 4: 3, 5: 2, 7: 0},
    'Ru': {2: 6, 3: 5, 4: 4},
    'Rh': {2: 7, 3: 6},
    'Pd': {2: 8, 4: 6},
    'Ag': {1: 10, 2: 9, 3: 8},
    'Cd': {2: 10},
    'Hf': {2: 2, 3: 1, 4: 0},
    'Ta': {3: 2, 4: 1, 5: 0},
    'W':  {2: 4, 3: 3, 4: 2, 5: 1, 6: 0},
    'Re': {2: 5, 3: 4, 4: 3, 5: 2, 7: 0},
    'Os': {2: 6, 3: 5, 4: 4},
    'Ir': {2: 7, 3: 6},
    'Pt': {2: 8, 4: 6},
    'Au': {1: 10, 2: 9, 3: 8},
    'Hg': {2: 10}
    }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~methods begin here
def get_d_count_luke(metal, ox_state):
    """
    Returns the d electron count of the metal center given its oxidation state

    Parameters
    ----------
    metal : str
        Atomic symbol of the transition metal (e.g. 'Fe', 'Co')
    ox_state : int
        Oxidation state of the metal (e.g. 2 for Fe²⁺)

    Returns
    -------
    int
        Number of d electrons

    Raises
    ------
    ValueError
        If metal or oxidation state not found in database
    """
    if metal not in d_electrons:
        raise ValueError(f"{metal} is not a transition metal!!")
    if ox_state not in d_electrons[metal]:
        raise ValueError(f"{ox_state} is not a valid oxidation state for {metal}!!")
    return d_electrons[metal][ox_state]

def get_ligand_charge(ligand):
    """
    Returns the charge of the inputted ligand (e.g. 'I-' returns -1, 'NH3' returns 0)

    Parameters
    ----------
    ligand : str
        Molecular formula of the ligand (e.g. 'I-', 'NH3')

    Returns
    -------
    int
        Charge of the ligand

    Raises
    ------
    ValueError
        If ligand name is not found in database
    """
    if ligand not in ligand_charges.keys():
        raise ValueError(f"{ligand} not found in ligand database :( ")
    return ligand_charges[ligand]

def find_spin(metal, d_count, ligands):
    """
    Returns the spin of the complex based on the type desired

    Parameters
    ----------
    spin : str
        How the spin of the complex will be determined, either by the system ('Auto') or by the user ('high', 'low')

    Returns
    -------
    int
        Spin of the complex times two, as used by the pyscf package

    Raises
    ------
    N/A
    """
    spin = get_spin_state(metal, d_count, ligands)
    if spin == 'low':
        return int(low_spins[d_count]*2)
    else:
        return int(high_spins[d_count]*2)                       ### May come back to later!!!!!!