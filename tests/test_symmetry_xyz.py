# tests/test_symmetry_xyz.py
import os
from pylft_mod.symmetry import (
    analyze_symmetry,
    get_sigma_salcs,
    get_pi_salcs,
    get_donor_indices_from_sym,
    print_symmetry_report,
    print_salc_report,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def run(name, xyz_file, ligand_list, expected_pg):
    path   = os.path.join(TEST_DIR, "xyz files", xyz_file)
    sym    = analyze_symmetry(path, ligand_list=ligand_list)
    pg     = sym["point_group"]
    status = "OK" if pg == expected_pg else "FAIL"
    print(f"  [{status}] {name:35s} detected: {pg:6s}  expected: {expected_pg}")
    return pg == expected_pg


def test_point_groups():
    print("\n" + "─"*60)
    print("  Point Group Detection Tests")
    print("─"*60)

    results = []

    # Oh complex no distortion

    # results.append(run(
    #     "Cr(CO)6              Oh",
    #     "crco6.xyz",
    #     ["CO"]*6,
    #     "Oh"
    # ))

    # Oh complex with weak JT distortion

    # results.append(run(
    #     "Fe(H2O)6  3+         Oh",
    #     "Fe_OH2_oct.xyz",
    #     ["OH2"]*6,
    #     "Oh"
    # ))

    # Octahedral complex with one ligand subtituted --> from Oh to C4v

    # results.append(run(
    #     "Cr(CO)5NH3           C4v",
    #     "cr(co)5nh3.xyz",
    #     ["CO"]*5 + ["NH3"],
    #     "C4v"
    # ))

    # Octahedral complex with 3 ligands subtituted in facial stereochemistry --> from Oh to C3v

    # results.append(run(
    #     "Cr(I-)3(F-)3           C3v",
    #     "Cr_Im_Fm_fac_oct.xyz",
    #     ["I-"]*3 + ["F-"]*3,
    #     "C3v"
    # ))

    # Octahedral complex with 3 ligands subtituted in meridional stereochemistry --> from Oh to C2v

    # results.append(run(
    #     "Fe(CN-)3(CO)3              C2v",
    #     "Fe_CmN_CmOp_mer_oct.xyz",
    #     ["[CN-]"]*3 + ["[CO]"]*3,
    #     "C2v"
    # ))

    print("─"*60)
    passed = sum(results)
    total  = len(results)
    print(f"  {passed}/{total} passed")
    print("─"*60 + "\n")


def test_salcs():
    print("\n" + "─"*60)
    print("  SALC Decomposition Tests")
    print("─"*60)

    # Cr(CO)6 -- Oh -- sigma: A1g + Eg + T1u, pi: T1g + T2g + T1u + T2u
    sym       = analyze_symmetry(
                    os.path.join(TEST_DIR, "crco6.xyz"),
                    ligand_list=["CO"]*6
                )
    print_symmetry_report(sym)

    donor_idx = get_donor_indices_from_sym(sym, donor_symbol="C")
    print(f"  Donor indices (C): {donor_idx}")

    sigma = get_sigma_salcs(sym, donor_idx)
    print_salc_report(sigma, "sigma")
    sigma_irreps = set(irr for irr, _ in sigma)
    assert "A_1g" in sigma_irreps, f"A_1g missing from sigma: {sigma_irreps}"
    assert "E_g"  in sigma_irreps, f"E_g missing from sigma: {sigma_irreps}"
    assert "T_1u" in sigma_irreps, f"T_1u missing from sigma: {sigma_irreps}"
    assert len(sigma) == 6,        f"Expected 6 sigma SALCs, got {len(sigma)}"
    print("  sigma SALCs: OK")

    pi = get_pi_salcs(sym, donor_idx)
    print_salc_report(pi, "pi")
    pi_irreps = set(irr for irr, _ in pi)
    assert "T_2g" in pi_irreps, f"T_2g missing from pi: {pi_irreps}"
    print("  pi SALCs: OK")

    print("─"*60 + "\n")


def test_error_handling():
    print("\n" + "─"*60)
    print("  Error Handling Tests")
    print("─"*60)

    # Wrong metal symbol
    try:
        from pylft_mod.symmetry import snap_to_octahedral
        from pylft_mod.molsym_local.molecule import Molecule
        mol = Molecule.from_file(os.path.join(TEST_DIR, "crco6.xyz"))
        snap_to_octahedral(mol, ligand_list=["CO"]*6)
        print("  [OK] snap runs without error on valid complex")
    except Exception as e:
        print(f"  [FAIL] unexpected error: {e}")

    # Invalid metal — not in table
    try:
        from pylft_mod.distortions import classify_distortion
        classify_distortion("Xx", 2, ["CO"]*6)
        print("  [FAIL] should have raised ValueError for unknown metal")
    except ValueError as e:
        print(f"  [OK] unknown metal caught: {e}")

    # d-count out of range
    try:
        from pylft_mod.distortions import classify_distortion
        classify_distortion("Sc", 10, ["CO"]*6)
        print("  [FAIL] should have raised ValueError for invalid d-count")
    except ValueError as e:
        print(f"  [OK] invalid d-count caught: {e}")

    print("─"*60 + "\n")


if __name__ == "__main__":
    test_point_groups()
    # test_salcs()
    # test_error_handling()