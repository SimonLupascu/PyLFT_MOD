
from pylft_mod.molsym_local import (
    Molecule, Symtext
)
from pylft_mod.molsym_local.symmetrize import symmetrize
import numpy as np
from pylft_mod.molsym_local.salcs.spherical_harmonics import SphericalHarmonics
from pylft_mod.molsym_local.salcs.projection_op import ProjectionOp

def analyze_symmetry(xyz_path: str) -> dict:
    """
    **Main function**

    Run full MolSym symmetry analysis on a XYZ file.
    The function will contain a tolerance parameter (in Angstrom)
    that dictates how far can an atom be from its symmetry-equivalent
    position before MolSym considers the symmetry broken

    Parameters
    ----------
    xyz_file : str
        Contains the coordinates of the complex analyzed

    Returns
    -------
    dict
        Dictionary of all symmetry data needed downstream.
    """

    mol      = Molecule.from_file(xyz_path)
    mol_sym  = symmetrize(mol, asym_tol=0.1)            # snap atoms to ideal positions
    st       = Symtext.from_molecule(mol_sym)    # now detects correctly

    return {
        "point_group":    str(st.pg),
        "order":          st.order,
        "irreps":         [i.symbol for i in st.irreps],
        "classes":        list(st.classes),
        "class_orders":   list(st.class_orders),
        "character_table": st.character_table,
        "symtext":        st,
        "mol":            mol_sym,   # use symmetrized mol downstream
    }


def get_sigma_salcs(sym_data: dict, donor_indices: list) -> list:
    """
    Build sigma-donor SALCs from a list of donor atom indices.

    Parameters
    ----------

    sym_data : dict
        Dictionary of symmetry data from *analyze_symmetry* function

    donor_indices : list
        List of ligand donor atoms in metal complex

            Example: idx = 0 => s-type sigma donor (l=0)

    Returns
    -------
    list
        List of (irrep_symbol, coefficients) tuples.
    """

    st      = sym_data["symtext"]
    mol     = sym_data["mol"]
    natoms  = len(mol.atoms)

    fxn_list = [[] for _ in range(natoms)]
    for idx in donor_indices:
        fxn_list[idx] = [0]

    fxn_set = SphericalHarmonics(st, fxn_list)
    salcs   = ProjectionOp(st, fxn_set)

    return [(s.irrep.symbol, s.coeffs) for s in salcs]


def get_pi_salcs(sym_data: dict, donor_indices: list) -> list:
    """
    Builds pi-donor/acceptor SALCs (l=1, p-type) from donor atom indices.

    Parameters
    ----------
    sym_data : str
        Symmetry data dictionary from *analyze symmetry* function

    donor_indices : list
        List of ligand donor atoms in metal complex

            Example: idx = 1 => p-type sigma donor (l=1)

    Returns
    -------
    list
        Tuple of (irrep_symbol, coefficients)
    """

    st      = sym_data["symtext"]
    mol     = sym_data["mol"]
    natoms  = len(mol.atoms)

    fxn_list = [[] for _ in range(natoms)]
    for idx in donor_indices:
        fxn_list[idx] = [1]    # l=1 = p-type pi

    fxn_set = SphericalHarmonics(st, fxn_list)
    salcs   = ProjectionOp(st, fxn_set)

    return [(s.irrep.symbol, s.coeffs) for s in salcs]


def print_symmetry_report(sym_data: dict) -> None:
    """
    Print a human-readable symmetry report to stdout.

    Parameters
    ----------
    sym_data : str
        Symmetry data dictionary from *analyze symmetry* function
    
    Returns
    -------
    None
        Prints the character table in Mulliken symbols in the user terminal
    """
    print(f"\n{'─'*50}")
    print(f"  Point group : {sym_data['point_group']}")
    print(f"  Group order : {sym_data['order']}")
    print(f"  Irreps      : {', '.join(sym_data['irreps'])}")
    print(f"\n  Character table:")
    print(f"  {'':8s}", end="")
    for cls in sym_data['classes']:
        print(f"  {cls:>6s}", end="")
    print()
    for i, irrep in enumerate(sym_data['irreps']):
        print(f"  {irrep:8s}", end="")
        for val in sym_data['character_table'][i]:
            print(f"  {val:>6.1f}", end="")
        print()
    print(f"{'─'*50}\n")


def print_salc_report(salcs: list, basis_type: str) -> None:
    """
    Print SALC decomposition;
    Which irreps appear and how many times.

    Parameters
    ----------
    salcs : list
        List of irreps containted in the Reductible Representation
    basis_type : str
        Irrep type contained in the Reductible Representation

    Returns
    -------
    None
        Prints the SALC decomposition from the Reductible Representation
    """
    from collections import Counter
    counts = Counter(irrep for irrep, _ in salcs)
    print(f"\n  {basis_type} SALC decomposition:")
    print(f"  G = " + " + ".join(
        f"{n}{irr}" if n>1 else irr
        for irr, n in sorted(counts.items())
    ))
    print(f"  Total SALCs: {len(salcs)}\n")
