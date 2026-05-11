import molsym
from pylft_mod.symmetry import (
    analyze_symmetry
)
# Define the weak field and strong field ligands
STRONG_FIELD = {"CO", "CN", "NO", "PR3"}
WEAK_FIELD   = {"H2O", "OH", "F", "Cl", "Br", "I"}

# Define the JT distortions as a fxn of d-count
JT_TABLE = {
    (4, "high"): "JT_strong",
    (7, "low"):  "JT_strong",
    (9, "any"):  "JT_strong",
    (1, "any"):  "JT_weak",
    (2, "any"):  "JT_weak",
    (4, "low"):  "JT_weak",
    (5, "low"):  "JT_weak",
    (7, "high"): "JT_weak",
}

#Define a list of d counts of all transition metals
D_COUNTS_NEUTRAL = {
    # 3d series
    "Sc": 3,  "Ti": 4,  "V":  5,  "Cr": 6,  "Mn": 7,
    "Fe": 8,  "Co": 9,  "Ni": 10, "Cu": 11, "Zn": 12,
    # 4d series
    "Y":  3,  "Zr": 4,  "Nb": 5,  "Mo": 6,  "Tc": 7,
    "Ru": 8,  "Rh": 9,  "Pd": 10, "Ag": 11, "Cd": 12,
    # 5d series
    "La": 3,  "Hf": 4,  "Ta": 5,  "W":  6,  "Re": 7,
    "Os": 8,  "Ir": 9,  "Pt": 10, "Au": 11, "Hg": 12,
}


def get_d_count(metal: str, ox_state: int) -> int:
    """
    Takes the metal and its oxidation state
    and returns the electron d count

    Parameters
    ----------
    metal : str
        Metal type in string format
    ox_state : int
        Oxidation state of the metal in the complex
    
    Return
    ------
    int
        Returns the electron d count of the metal in the complex

    Examples
    --------
    Cr, 0 -> 6;

    Cu, 2 -> 9;
    """
    neutral = D_COUNTS_NEUTRAL.get(metal)
    if neutral is None:
        raise ValueError(f"Metal {metal} not in d-count table")
    d = neutral - ox_state
    if not 0 <= d <= 10:
        raise ValueError(f"Metal has an invalid d count of {d}")
    return d


def get_spin_state(metal: str, d_count: int, ligands: list) -> str:
    """
    Determines spin state from ligand field strength and metal row.
    4d/5d metals are always low spin regardless of ligand.
    3d metals depend on ligand field strength.

    Parameters
    ----------
    metal : str
        Metal type in string format
    d_count : int
        Electron d count of metal in the complex
    ligands : list
        List of ligands participating in the distortion

    Return
    ------
    str
        String containing the spin state of the metal complex
    """
    FOURTH_ROW = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}
    FIFTH_ROW  = {"La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
    
    if metal in FOURTH_ROW or metal in FIFTH_ROW:
        return "low"
    
    ligand_set = set(ligands)
    if ligand_set & STRONG_FIELD:
        return "low"
    elif ligand_set & WEAK_FIELD:
        return "high"
    else:
        return "low" #assumed intermediate ligands would form low spin complex


def classify_distortion (metal: str, ox_state: int, ligands: list) -> dict:
    """
    Main function — determines distortion type
    for an Oh complex.

    Parameters
    ----------
    metal : str
        Metal type in string format
    ox_state : int
        Oxidation state of the metal in the complex
    ligands : list
        List of ligands participating in the distortion

    Return
    ------
    dict
        Returns a dict with:
        d_count       : int
        spin          : str  "high" or "low"
        distortion    : str  "JT_strong", "JT_weak", "pi_backbonding", "none"
        target_group  : str  the subgroup to reduce to, or None
        notes         : str  Explanation of distortion, if it exits
    """
    d_count = get_d_count(metal, ox_state)
    spin = get_spin_state(metal, d_count, ligands)

    distortion = JT_TABLE.get(
        (d_count, spin), JT_TABLE.get((d_count, "any"), "none")
    )

    has_pi_acceptor = bool(set(ligands) & STRONG_FIELD)
    has_pi_donor = bool(set(ligands) & WEAK_FIELD)
    if distortion == "none" and (has_pi_acceptor or has_pi_donor):
        distortion = "pi_backbonding"

    TARGET_GROUP = {
        "JT_strong" : "D4h",
        "JT_weak" : "D3d",
        "pi_backbonding" : None,
        "none" : None
    }
    
    NOTES = {
        "JT_strong":        f"d{d_count} {spin} spin - Eg degeneracy drives tetragonal distortion to D4h",
        "JT_weak":          f"d{d_count} {spin} spin - T2g degeneracy, weak trigonal distortion",
        "pi_backbonding":   f"d{d_count} - No JT distortion, but pi interactions present, T2g affected",
        "none":             f"d{d_count} - No distortion expected after calculations"
    }

    return {
        "d_count":      d_count,
        "spin":         spin,
        "distortion":   distortion,
        "target_group": TARGET_GROUP[distortion],
        "notes":        NOTES[distortion]
    }