import os
from pylft_mod.symmetry import(
    analyze_symmetry,
    get_sigma_salcs,
    get_pi_salcs,
    print_symmetry_report,
    print_salc_report,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
XYZ_PATH = os.path.join(TEST_DIR, "crco6.xyz")

def test_crco6():

    sym = analyze_symmetry(XYZ_PATH)
    print_symmetry_report(sym)

    assert sym["point_group"] == "Oh",  f"Something went wrong with the POINT GROUP"
    assert sym["order"] == 48,          f"Something went wrong with the ORDER"
    assert "T_2g" in sym["irreps"],     f"T_2g is not in the irrep !"
    assert "E_g"  in sym["irreps"],     f"E_g is not in the irrep !"
    assert "A_1g" in sym["irreps"],     f"A_1g is not the irrep !"

    donor_idx = [1, 2, 3, 4, 5, 6]

    sigma = get_sigma_salcs(sym, donor_idx)
    print_salc_report(sigma, "sigma")
    sigma_irreps = set(irr for irr, _ in sigma)
    assert "A_1g" in sigma_irreps
    assert "E_g"  in sigma_irreps
    assert "T_1u" in sigma_irreps
    assert len(sigma) == 6

    pi = get_pi_salcs(sym, donor_idx)
    print_salc_report(pi, "pi")
    pi_irreps = set(irr for irr, _ in pi)
    assert "T_2g" in pi_irreps,         f"No backbonding in the complex"

    print("All checks passed")
    
if __name__ == "__main__":
    test_crco6()