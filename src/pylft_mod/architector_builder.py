import numpy as np
from architector import build_complex, convert_io_molecule
import os, glob
from rdkit import Chem
from pylft_mod.distortions import classify_distortion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Delete any old .xyz and .mol2 files before building new ones --- SOS!
for old_file in glob.glob(os.path.join(BASE_DIR, "*.xyz")):
    os.remove(old_file)

for old_file in glob.glob(os.path.join(BASE_DIR, "*.mol2")):
    os.remove(old_file)

# Clarifying some ligand donor exceptions!
donor_exceptions = {
    "[C-]#N":    0,   # bonds through C, not N
    "[C-]#[O+]": 0,   # bonds through C, not O
    "[N]=O":     0,   # bonds through N
}

#Architector way to enforce geometry --- see README coordList.
isomer_coord_vectors = {
    "trans": [
        [ 0.0,  0.0,  2.0],  
        [ 0.0,  0.0, -2.0],
        [ 2.0,  0.0,  0.0],
        [-2.0,  0.0,  0.0],
        [ 0.0,  2.0,  0.0],
        [ 0.0, -2.0,  0.0],
    ],
    "cis": [
        [ 2.0,  0.0,  0.0],
        [ 0.0,  2.0,  0.0],
        [-2.0,  0.0,  0.0],
        [ 0.0, -2.0,  0.0],
        [ 0.0,  0.0,  2.0],
        [ 0.0,  0.0, -2.0],
    ],
    "fac": [
        [ 2.0,  0.0,  0.0],
        [ 0.0,  2.0,  0.0],
        [ 0.0,  0.0,  2.0],
        [-2.0,  0.0,  0.0],
        [ 0.0, -2.0,  0.0],
        [ 0.0,  0.0, -2.0],
    ],
    "mer": [
        [ 2.0,  0.0,  0.0],
        [-2.0,  0.0,  0.0],
        [ 0.0,  2.0,  0.0],
        [ 0.0, -2.0,  0.0],
        [ 0.0,  0.0,  2.0],
        [ 0.0,  0.0, -2.0],
    ],
}

ligand_charge = {
    "[C-]#[O+]": 0,
    "[C-]#N": -1,
    "[N]=O": 0,
    "CP(C)C": 0,
    "[OH2]": 0,
    "[OH-]": -1,
    "[F-]": -1,
    "[Cl-]": -1,
    "[Br-]": -1,
    "[I-]": -1,
    "[NH3]": 0
}

smiles_to_name = {
    "[C-]#[O+]": "CO",
    "[C-]#N": "CN-",
    "[N]=O": "NO",
    "CP(C)C": "PR3",
    "[OH2]": "OH2",
    "[OH-]": "OH-",
    "[F-]": "F-",
    "[Cl-]": "Cl-",
    "[Br-]": "Br-",
    "[I-]": "I-",
    "[NH3]": "NH3",
    "N": "NH3"
}

def calculate_complex_charge(metal_oxidation, ligands_used):

    total_ligand_charge = 0 
    for smiles in ligands_used:
        #use initial charge as 0 if not of the ones stated in ligand_charge
        charge = ligand_charge.get(smiles, 0) 
        total_ligand_charge += charge
    
    total_charge = metal_oxidation + total_ligand_charge
    return total_charge

# Definition of each ligand atom's index numbers --- IMPORTANT, otherwise for ligands like
# P(Me)3 Architector cannot assign a coordination number itself and the code crashes!
def find_donor_atom(smiles: str) -> int:
    # Finds the index of the ligand atom that bonds with the metal (donor atom)

    # Check if the ligand exists in the donor exceptions dictionary
    if smiles in donor_exceptions:
        return donor_exceptions[smiles]

    # Priority rule: P > N > O > S > C
    donor_priority = {"P": 0, "N": 1, "O": 2, "S": 3, "C": 4}

    molecule      = Chem.MolFromSmiles(smiles)
    best_index    = 0
    best_priority = 999  # start high so any real atom beats it

    for atom in molecule.GetAtoms():
        symbol   = atom.GetSymbol()
        priority = donor_priority.get(symbol, 99)  # 99 if not in the dict

        if priority < best_priority:
            best_priority = priority
            best_index    = atom.GetIdx()

    return best_index

def get_unpaired_electrons(d_count, spin):
    # Number of unpaired electrons based on d-count and spin state
    if spin == "high":
        return d_count if d_count <= 5 else 10 - d_count
    else:  # low spin
        if d_count <= 3:
            return d_count
        elif d_count <= 6:
            return 6 - d_count
        elif d_count <= 8:
            return d_count - 6
        else:
            return 10 - d_count

def octahedral_complex(metal, ligand, oxidation_state):
    donor_atom = find_donor_atom(ligand)
    total_charge = calculate_complex_charge(oxidation_state, [ligand] * 6)

    ligand_name = smiles_to_name.get(ligand, ligand)
    info        = classify_distortion(metal, oxidation_state, [ligand_name] * 6)
    metal_spin  = get_unpaired_electrons(info["d_count"], info["spin"])

    print(f"  Spin state: {info['spin']} | Unpaired electrons: {metal_spin}")
    print(f"  Distortion: {info['distortion']} | {info['notes']}")

    my_input = {
        "core": {
            "metal":    metal,
            "coreType": "octahedral",
        },
        # DONT MESS UP THE SMILES NOTATION -- CODE CRASHES OTHERWISE!
        # Ligand properties are repeated once for each of the 6 coordination sites!
        "ligands": [
            {"smiles": ligand, "ligType": "mono", "coordList": [donor_atom]},
            {"smiles": ligand, "ligType": "mono", "coordList": [donor_atom]},
            {"smiles": ligand, "ligType": "mono", "coordList": [donor_atom]},
            {"smiles": ligand, "ligType": "mono", "coordList": [donor_atom]},
            {"smiles": ligand, "ligType": "mono", "coordList": [donor_atom]},
            {"smiles": ligand, "ligType": "mono", "coordList": [donor_atom]},
        ],
        "parameters": {
            "metal_ox": oxidation_state,
            "metal_spin": metal_spin,
            "return_only_1": True,
            "relax": True,
            "force_generation": True,  # forces construction to proceed even if difficult
            "full_method": "GFN2-xTB"
        }
    }

    print(f"\nBuilding [{metal}({ligand})6] | metal ox: +{oxidation_state} | complex charge: {total_charge} ...")

    out = build_complex(my_input)

    key               = list(out.keys())[0]
    complex_structure = out[key]
    print(f"Done. Key: {key}")

    return complex_structure, key


def heteroleptic_complex(metal, oxidation_state, ligand_a, ligand_b, isomer):
    donor_a = find_donor_atom(ligand_a)
    donor_b = find_donor_atom(ligand_b)

    name_a = smiles_to_name.get(ligand_a, ligand_a)
    name_b = smiles_to_name.get(ligand_b, ligand_b)
    info   = classify_distortion(metal, oxidation_state, [name_a, name_b])
    metal_spin = get_unpaired_electrons(info["d_count"], info["spin"])

    print(f"  Spin state: {info['spin']} | Unpaired electrons: {metal_spin}")
    print(f"  Distortion: {info['distortion']} | {info['notes']}")

    if isomer == "none":
        # Let Architector decide the geometry freely
        core_definition = {"metal": metal, "coreType": "octahedral"}
        count_a = 2
        count_b = 4
        ligands = (
            [{"smiles": ligand_a, "ligType": "mono", "coordList": [donor_a]}] * count_a +
            [{"smiles": ligand_b, "ligType": "mono", "coordList": [donor_b]}] * count_b
        )

    else:
        #Architector README.
        vectors = isomer_coord_vectors[isomer]
        count_a = 2 if isomer in ("cis", "trans") else 3
        count_b = 6 - count_a


    ligands_used = [ligand_a] * count_a + [ligand_b] * count_b
    total_charge = calculate_complex_charge(oxidation_state, ligands_used)

    if total_charge < -2:
        relax_setting = False
        method        = "GFN-FF"
        print(f"  Note: complex charge is {total_charge}, using GFN-FF (xTB cannot converge on this).")
    else:
        relax_setting = True
        method        = "GFN2-xTB"

        ligands = []
        for i in range(count_a):
            ligands.append({
                "smiles":    ligand_a,
                "ligType":   "mono",
                "coordList": [[donor_a, i]],
            })
        for i in range(count_b):
            ligands.append({
                "smiles":    ligand_b,
                "ligType":   "mono",
                "coordList": [[donor_b, count_a + i]],
            })

        core_definition = {
            "metal":     metal,
            "coordList": vectors,   # hand Architector the exact 3D positions
        }

    print(f"\nBuilding [{metal}({ligand_a})_{count_a}({ligand_b})_{count_b}] | isomer: {isomer} | metal ox: +{oxidation_state} | complex charge: {total_charge} ...")

    my_input = {
        "core":    core_definition,
        "ligands": ligands,
        "parameters": {
            "metal_ox": oxidation_state,
            "metal_spin": metal_spin,
            "return_only_1": True,
            "relax": True,
            "force_generation": True,
            "full_method": "GFN2-xTB",
            "n_symmetries": 10,
        }
    }

    out = build_complex(my_input)

    if not out:
        raise ValueError("Architector returned no structures. Try a different isomer or ligand.")

    key               = list(out.keys())[0]
    complex_structure = out[key]
    print(f"Done. Key: {key}")

    return complex_structure, key


def symbols_and_positions(complex_structure):
    moles     = convert_io_molecule(complex_structure["mol2string"])
    symbols   = moles.ase_atoms.get_chemical_symbols()
    positions = moles.ase_atoms.get_positions()  # (N, 3) array in Angstroms units!!

    return symbols, positions, moles


def xyz_coordinates(symbols, positions):
    center             = positions[0]
    centered_positions = positions - center

    #Check stack overflow bibliography
    header = f"{'Atom':<4} {'X':>12} {'Y':>12} {'Z':>12}"
    print(header)
    print("-" * 40)

    for sym, pos in zip(symbols, centered_positions):
        print(f"{sym:<4} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")


def save_file(symbols, positions, filename):
    # Write a metal-centred XYZ file
    center             = positions[0]
    centered_positions = positions - center

    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write("Requested Octahedral Complex\n")
        for sym, p in zip(symbols, centered_positions):
            f.write(f"{sym}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n")

    print(f"Saved: {filename}")


def save_mol2(complex_structure, filename):

    mol2_content = complex_structure["mol2string"]
    with open(filename, "w") as f:
        f.write(mol2_content)
    print(f"Saved: {filename}")


def make_filename(metal, ligand_a, ligand_b, isomer):

    def clean(smiles):
        smiles = smiles.replace("[", "")
        smiles = smiles.replace("]", "")
        smiles = smiles.replace("-", "m")
        smiles = smiles.replace("+", "p")
        smiles = smiles.replace("#", "")
        smiles = smiles.replace("@", "")
        return smiles

    clean_a = clean(ligand_a)

    if ligand_b is None:
        # Homoleptic: only one ligand, no isomer in the filename
        filename = f"{metal}_{clean_a}_oct.xyz"
    else:
        # Heteroleptic: include both ligands and the isomer
        clean_b  = clean(ligand_b)
        filename = f"{metal}_{clean_a}_{clean_b}_{isomer}_oct.xyz"

    return os.path.join(BASE_DIR, filename)


def prompt_molecule():
    print("      Homoleptic Octahedral Complex Builder")
    print("\nEnter any metal symbol from the periodic table (Fe, Co, Cr, Mn, Ni, Cu, Zn, Ru, Rh, etc.):")
    print("Note: noble gases and pure nonmetals will fail!")
    print("\nVALID LIGANDS (brackets [ ] are REQUIRED)")
    print("Strong Field Ligands:")
    print("  [C-]#[O+]  carbonyl")
    print("  [C-]#N     cyanide")
    print("  [N]=O      nitric oxide")
    print("  CP(C)C     trimethylphosphine")
    print("\nWeak Field Ligands:")
    print("  [OH2]      water")
    print("  [OH-]      hydroxide")
    print("  [F-]       fluoride")
    print("  [Cl-]      chloride")
    print("  [Br-]      bromide")
    print("  [I-]       iodide")
    print("  [NH3]      ammonia")

    # .capitalize() prevents Architector from crashing on lowercase input --- SOS!
    metal           = input("\nEnter metal symbol: ").strip().capitalize()
    ligand          = input("Enter ligand: ").strip()
    oxidation_state = int(input("Enter oxidation state: ").strip())

    return metal, ligand, oxidation_state


def prompt_heteroleptic():
    print("\nHeteroleptic Mode")
    print("\nEnter any metal symbol from the periodic table (Fe, Co, Cr, Mn, Ni, Cu, Zn, Ru, Rh, etc.):")
    print("Note: noble gases and pure nonmetals will fail!")
    print("\nVALID LIGANDS (brackets [ ] are REQUIRED)")
    print("Strong Field Ligands:")
    print("  [C-]#[O+]  carbonyl")
    print("  [C-]#N     cyanide")
    print("  [N]=O      nitric oxide")
    print("  CP(C)C     trimethylphosphine")
    print("\nWeak Field Ligands:")
    print("  [OH2]      water")
    print("  [OH-]      hydroxide")
    print("  [F-]       fluoride")
    print("  [Cl-]      chloride")
    print("  [Br-]      bromide")
    print("  [I-]       iodide")
    print("  [NH3]      ammonia")
    print("You will choose two different ligands A and B.")
    print("Together they must fill all 6 sites of the octahedron.")
    print("\nAvailable isomers:")
    print("  cis   -> 2 of ligand A + 4 of ligand B, A ligands 90 degrees apart")
    print("  trans -> 2 of ligand A + 4 of ligand B, A ligands 180 degrees apart")
    print("  fac   -> 3 of ligand A + 3 of ligand B, A ligands on one triangular face")
    print("  mer   -> 3 of ligand A + 3 of ligand B, A ligands in a straight row")
    print("  none  -> Architector decides the geometry freely")
    print("")

    metal    = input("Enter metal symbol: ").strip().capitalize()
    ligand_a = input("Enter ligand A SMILES: ").strip()
    ligand_b = input("Enter ligand B SMILES: ").strip()
    isomer   = input("Enter isomer (cis / trans / fac / mer / none): ").strip().lower()

    valid_isomers = list(isomer_coord_vectors.keys()) + ["none"]
    if isomer not in valid_isomers:
        print(f"  '{isomer}' not recognised. Defaulting to none.")
        isomer = "none"

    oxidation_state = int(input("Enter oxidation state: ").strip())

    return metal, ligand_a, ligand_b, oxidation_state, isomer


def main():
    while True:
        print("\n  Octahedral Complex Builder")
        print("  1 - Homoleptic   (6 identical ligands)")
        print("  2 - Heteroleptic (two different ligands)")

        mode = input("Choose mode (1 or 2): ").strip()

        if mode == "1":
            metal, ligand, oxidation_state = prompt_molecule()

            try:
                complex_structure, key = octahedral_complex(metal, ligand, oxidation_state)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\nError: could not build complex.")
                print(f"Technical detail: {e}")
                again = input("Try again? (y/n): ").strip().lower()
                if again != "y":
                    break
                continue

            filename = make_filename(metal, ligand, None, None)

        elif mode == "2":
            metal, ligand_a, ligand_b, oxidation_state, isomer = prompt_heteroleptic()

            try:
                complex_structure, key = heteroleptic_complex(
                    metal, oxidation_state, ligand_a, ligand_b, isomer)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\nError: could not build complex.")
                print(f"Technical detail: {e}")
                again = input("Try again? (y/n): ").strip().lower()
                if again != "y":
                    break
                continue

            filename = make_filename(metal, ligand_a, ligand_b, isomer)

        else:
            print("Please enter 1 or 2.")
            continue

        symbols, positions, moles = symbols_and_positions(complex_structure)
        xyz_coordinates(symbols, positions)
        filename_mol2 = filename.replace(".xyz", ".mol2")
        save_file(symbols, positions, filename)
        save_mol2(complex_structure, filename_mol2)

        again = input("\nBuild another complex? (y/n): ").strip().lower()
        if again != "y":
            print("\nDone. Open visualize.ipynb to view your molecules.")
            break


if __name__ == "__main__":
    main()