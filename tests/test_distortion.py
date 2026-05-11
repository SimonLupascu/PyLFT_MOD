from pylft_mod.distortions import(
    classify_distortion
)

def test_classify_distortion():
    # Cr(CO)6 — d6 low spin, no JT, CO is pi acceptor
    r = classify_distortion("Cr", 0, ["CO"]*6)
    assert r["d_count"]    == 6,               f"Expected d6, got d{r['d_count']}"
    assert r["spin"]       == "low",           f"Expected low spin, got {r['spin']}"
    assert r["distortion"] == "pi_backbonding",f"Expected pi-backbonding, got {r['distortion']}"
    assert r["target_group"] is None,          f"Expected no subgroup, got {r['target_group']}"
    print(f"Cr(CO)6  -> d{r['d_count']} {r['spin']} spin | {r['distortion']} | {r['notes']}")

    # Cu(NH3)6 2+ — d9, always JT_strong regardless of ligand
    r = classify_distortion("Cu", 2, ["NH3"]*6)
    assert r["d_count"]    == 9,               f"Expected d9, got d{r['d_count']}"
    assert r["distortion"] == "JT_strong",     f"Expected JT-strong, got {r['distortion']}"
    assert r["target_group"] == "D4h",         f"Expected D4h, got {r['target_group']}"
    print(f"Cu(NH3)6 -> d{r['d_count']} {r['spin']} spin | {r['distortion']} | {r['notes']}")

    print("\nAll distortion tests passed")

if __name__ == "__main__":
    test_classify_distortion()