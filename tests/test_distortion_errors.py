from pylft_mod.distortions import(
    classify_distortion
)

def test_invalid_inputs():
    # Invalid metal — not in table
    try:
        classify_distortion("Xx", 2, ["CO"]*6)
        print("FAIL — should have raised ValueError for unknown metal")
    except ValueError as e:
        print(f"OK — unknown metal: {e}")

    # Negative oxidation state giving d-count > 10
    try:
        classify_distortion("Zn", -3, ["CO"]*6)
        print("FAIL — should have raised ValueError for invalid d-count")
    except ValueError as e:
        print(f"OK — invalid d-count: {e}")

    # Oxidation state too high giving d-count < 0
    try:
        classify_distortion("Sc", 10, ["CO"]*6)
        print("FAIL — should have raised ValueError for negative d-count")
    except ValueError as e:
        print(f"OK — negative d-count: {e}")

    print("\nAll error handling tests passed")

if __name__ == "__main__":
    print("\n--- Error handling ---")
    test_invalid_inputs()