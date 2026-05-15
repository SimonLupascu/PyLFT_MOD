
from pylft_mod.molsym_local import (
    Molecule, Symtext
)
from pylft_mod.molsym_local.symmetrize import symmetrize
import numpy as np
from pylft_mod.molsym_local.salcs.spherical_harmonics import SphericalHarmonics
from pylft_mod.molsym_local.salcs.projection_op import ProjectionOp

DONOR_ATOM = {
    # Strong field
    "CO":   "C",    # carbon binds to metal
    "CN-":  "C",    # carbon end binds
    "NO":   "N",
    "PR3":  "P",
    # Weak field
    "H2O":  "O",
    "OH2":  "O",
    "OH-":  "O",
    "F-":   "F",
    "Cl-":  "Cl",
    "Br-":  "Br",
    "I-":   "I",
    "NH3":  "N",
}

ALL_LIGANDS = {**{k: v for k,v in DONOR_ATOM.items()}}

def snap_to_octahedral(mol, ligand_list: list = None):
    """
    Snaps donor atoms to idealized octahedral positions.
    
    If ligand_list is provided, uses DONOR_ATOM table to identify
    donor atoms unambiguously. Otherwise falls back to distance-based
    detection.

    Parameters
    ----------
    mol : Molecule
        Molecule object containing the complex to be snapped
    ligand_list : list, optional
        List of ligand names (e.g. ["H2O"]*6) to identify donor atoms, by default None
    
    Returns
    -------
    Molecule
        New Molecule object with donor atoms snapped to ideal octahedral positions
    """

    coords = mol.coords.copy()
    atoms  = [str(s) for s in mol.atoms]

    # Find metal
    ALL_NONMETAL = {"O","N","C","H","P","S","F","Cl","Br","I"}
    metal_idx = next(i for i,s in enumerate(atoms) if s not in ALL_NONMETAL)
    metal_pos = coords[metal_idx]

    if ligand_list:
        donor_symbols = set(
            DONOR_ATOM.get(lig) for lig in ligand_list
            if DONOR_ATOM.get(lig) is not None
        )

        # Get all atoms with correct donor symbol, sort by distance
        # unit-agnostic — works in both Angstrom and Bohr
        candidates = [
            (i, np.linalg.norm(coords[i] - metal_pos))
            for i, sym in enumerate(atoms)
            if sym in donor_symbols and i != metal_idx
        ]
        candidates.sort(key=lambda x: x[1])
        donor_indices = [i for i, d in candidates[:len(ligand_list)]]
        print(f"  DEBUG: donors={[atoms[i] for i in donor_indices]} "
              f"at distances={[round(d,3) for _,d in candidates[:len(ligand_list)]]}")

    else:
        # Fallback: N closest non-H non-metal atoms
        candidates = [
            (i, np.linalg.norm(coords[i] - metal_pos))
            for i, sym in enumerate(atoms)
            if sym != "H" and i != metal_idx
        ]
        candidates.sort(key=lambda x: x[1])
        donor_indices = [i for i, d in candidates[:6]]

    if not donor_indices:
        print("  WARNING: no donor atoms found, skipping snap")
        return mol

    from collections import defaultdict
    symbol_groups = defaultdict(list)
    for i in donor_indices:
        symbol_groups[atoms[i]].append(i)

    for sym, indices in symbol_groups.items():
        avg = np.mean([np.linalg.norm(coords[i] - metal_pos) for i in indices])
        for i in indices:
            direction = coords[i] - metal_pos
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                coords[i] = metal_pos + (direction / norm) * avg
        print(f"  Snapped {len(indices)} {sym} donors to r={avg:.3f}")

    # For 6 donors: snap to ideal ±x, ±y, ±z axes ~~~~~~~~~~~HARDCODE FIX FOR OCTAHEDRAL SYMMETRY
    if len(donor_indices) == 6:
        OH_VECTORS = np.array([
            [ 1, 0, 0], [-1, 0, 0],
            [ 0, 1, 0], [ 0,-1, 0],
            [ 0, 0, 1], [ 0, 0,-1],
        ], dtype=float)

        used = []
        for i in donor_indices:
            direction = coords[i] - metal_pos
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            unit = direction / norm
            # find closest unused ideal vector
            dots = [np.dot(unit, v) for v in OH_VECTORS]
            for ranked in np.argsort(dots)[::-1]:
                v = tuple(OH_VECTORS[ranked])
                if v not in used:
                    coords[i] = metal_pos + OH_VECTORS[ranked] * avg
                    used.append(v)
                    break

    # For other CN: snap each donor group to its own average distance only
    else:
        from collections import defaultdict
        symbol_groups = defaultdict(list)
        for i in donor_indices:
            symbol_groups[atoms[i]].append(i)
        for sym, indices in symbol_groups.items():
            avg = np.mean([np.linalg.norm(coords[i] - metal_pos)
                           for i in indices])
            for i in indices:
                direction = coords[i] - metal_pos
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    coords[i] = metal_pos + (direction / norm) * avg
            print(f"  Snapped {len(indices)} {sym} donors to r={avg:.3f}")

    for i, sym in enumerate(atoms):
        if sym == "H":
            # find closest donor atom as parent
            donor_dists = [(j, np.linalg.norm(coords[i] - coords[j]))
                           for j in donor_indices]
            parent_idx  = min(donor_dists, key=lambda x: x[1])[0]

            # get original H-donor vector and preserve it
            h_vec  = coords[i] - coords[parent_idx]
            coords[i] = coords[parent_idx] + h_vec

    # H atoms on H2O and NH3 break symmetrize — they are irrelevant
    # to the coordination sphere point group, so we can just remove them for the symmetry analysis.

    keep    = [i for i, sym in enumerate(atoms) if sym != "H"]
    atoms   = [atoms[i]    for i in keep]
    coords  = coords[keep]
    masses  = [mol.masses[i] for i in keep]

    return Molecule(atoms, coords, masses)

def analyze_symmetry(xyz_path: str, ligand_list: list = None) -> dict:
    """
    **Main function**

    Run full MolSym symmetry analysis on a XYZ file.
    The function will contain a tolerance parameter (in Angstrom)
    that dictates how far can an atom be from its symmetry-equivalent
    position before MolSym considers the symmetry broken.

    Parameters
    ----------
    xyz_file : str
        Contains the coordinates of the complex analyzed in XYZ format.
        The first line is the number of atoms, the second line is a comment,
        and the following lines contain the element symbol and x, y, z coordinates for each atom.
    ligand_list : list, optional
        List of ligand names (e.g. ["H2O"]*6) to identify donor atoms, by default None

    Returns
    -------
    dict
        Dictionary of all symmetry data needed for SALC analysis, including:
        - point group
        - group order
        - list of irreps
        - list of classes
        - list of class orders
        - character table
    """

    mol = Molecule.from_file(xyz_path)
    mol = snap_to_octahedral(mol, ligand_list=ligand_list)

    # Try symmetrize first, fall back to direct detection with loose tolarance
    st  = None
    pg  = None

    for tol in [0.1, 0.3, 0.5, 0.8]:
        try:
            mol_sym = symmetrize(mol, asym_tol=tol)
            st      = Symtext.from_molecule(mol_sym)
            pg      = str(st.pg)
            if pg not in ("C1", "Ci") or tol == 0.8:
                if tol > 0.1:
                    print(f"  NOTE: needed asym_tol={tol} to detect {pg}")
                break
        except Exception as e:
            print(f"  Error: symmetrize failed at tol={tol}: {e}")
            continue

    # Fallback: skip symmetrize, use direct detection with loose tol
    if st is None or pg in ("C1", "Ci"):
        print("  NOTE: symmetrize failed, trying direct detection")
        for tol in [0.1, 0.3, 0.5, 0.8]:
            try:
                mol.tol = tol
                st      = Symtext.from_molecule(mol)
                pg      = str(st.pg)
                mol_sym = mol
                if pg not in ("C1", "Ci") or tol == 0.8:
                    print(f"  NOTE: direct detection at tol={tol} -> {pg}")
                    break
            except Exception as e:
                print(f"  Error: direct detection failed at tol={tol}: {e}")
                continue

    if st is None:
        raise RuntimeError(f"Could not detect point group for {xyz_path}")

    return {
        "point_group":    pg,
        "order":          st.order,
        "irreps":         [i.symbol for i in st.irreps],
        "classes":        list(st.classes),
        "class_orders":   list(st.class_orders),
        "character_table": st.character_table,
        "symtext":        st,
        "mol":            mol_sym,
    }


def get_sigma_salcs(sym_data: dict, donor_indices: list) -> list:
    """
    Builds sigma-donor/acceptor SALCs (l=0, s-type) from donor atom indices.

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
        List of (irrep_symbol, coefficients) tuples for each SALC, where:
        - irrep_symbol is the Mulliken symbol of the irrep
        - coefficients is a list of the SALC coefficients for each donor atom
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

def get_donor_indices_from_sym(sym_data: dict, donor_symbol: str) -> list:
    """Get donor indices from the symmetry-analyzed (H-stripped) molecule.
    
    Parameters
    ----------
    sym_data : dict
        Dictionary of symmetry data from *analyze_symmetry* function
    donor_symbol : str
        Symbol of the donor atom type to find indices for

    Returns
    -------
    list
        List of indices for the specified donor atoms
    """
    mol    = sym_data["mol"]
    atoms  = [str(s) for s in mol.atoms]
    metal  = next(s for s in atoms if s not in 
                  {"O","N","C","H","P","S","F","Cl","Br","I"})
    metal_idx = atoms.index(metal)
    metal_pos = mol.coords[metal_idx]
    
    candidates = [
        (i, np.linalg.norm(mol.coords[i] - metal_pos))
        for i, sym in enumerate(atoms)
        if sym == donor_symbol and i != metal_idx
    ]
    candidates.sort(key=lambda x: x[1])
    return [i for i, _ in candidates]

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
