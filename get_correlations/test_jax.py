"""
Test script to verify that the JAX version of get_correlations
returns the same results as the NumPy version.
"""
import numpy as np
import time
from Correlator import Correlator

def test_jax_correlations(map_path='../../Drive/Simulations/SkyModels/ULSA_maps/200.fits', 
                          mu_bin1=0, mu_bin2=0):
    """
    Test that JAX version returns same results as NumPy version.
    
    Parameters
    ----------
    map_path : str
        Path to the FITS map file
    mu_bin1 : int
        First mu bin index (0 is fastest for testing)
    mu_bin2 : int
        Second mu bin index (0 is fastest for testing)
    
    Returns
    -------
    bool
        True if tests pass, False otherwise
    """
    print(f"Testing with mu_bin1={mu_bin1}, mu_bin2={mu_bin2}")
    print("-" * 50)
    
    # Create correlator
    corr = Correlator(map_path, mu_bin1, mu_bin2)
    print(f"N1 (bin1 pixels): {len(corr.bin1_ndx)}")
    print(f"N2 (bin2 pixels): {len(corr.bin2_ndx)}")
    print(f"nFreq: {corr.nFreq}")
    print(f"nD (distance bins): {corr.nD}")
    
    # Run NumPy version
    print("\nRunning NumPy version...")
    t0 = time.time()
    cors_np, counts_np = corr.get_correlations()
    t_np = time.time() - t0
    print(f"NumPy time: {t_np:.3f} seconds")
    
    # Run JAX version (fast)
    print("\nRunning JAX version (first call includes JIT compilation)...")
    t0 = time.time()
    cors_jax, counts_jax = corr.get_correlations_jax()
    t_jax_first = time.time() - t0
    print(f"JAX time (with JIT): {t_jax_first:.3f} seconds")
    
    # Run JAX version again (compiled)
    print("\nRunning JAX version (second call, JIT compiled)...")
    t0 = time.time()
    cors_jax2, counts_jax2 = corr.get_correlations_jax()
    t_jax_compiled = time.time() - t0
    print(f"JAX time (compiled): {t_jax_compiled:.3f} seconds")
    
    # Compare results
    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    
    # Check counts
    counts_match = np.allclose(counts_np, counts_jax, rtol=1e-5)
    print(f"\nCounts match: {counts_match}")
    if not counts_match:
        print(f"  Max counts difference: {np.max(np.abs(counts_np - counts_jax))}")
        print(f"  NumPy counts: {counts_np}")
        print(f"  JAX counts: {counts_jax}")
    
    # Check correlations
    # Note: We use a larger rtol because we're dealing with very large numbers
    # and the reduction order differs between NumPy loops and JAX vectorized ops
    cors_match = np.allclose(cors_np, cors_jax, rtol=1e-4, atol=1e-10)
    print(f"Correlations match (rtol=1e-4): {cors_match}")
    if not cors_match:
        max_diff = np.max(np.abs(cors_np - cors_jax))
        max_rel_diff = np.max(np.abs(cors_np - cors_jax) / (np.abs(cors_np) + 1e-10))
        print(f"  Max absolute difference: {max_diff}")
        print(f"  Max relative difference: {max_rel_diff}")
        
        # Find where the differences are largest
        diff = np.abs(cors_np - cors_jax)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Largest diff at index: {max_idx}")
        print(f"  NumPy value: {cors_np[max_idx]}")
        print(f"  JAX value: {cors_jax[max_idx]}")
    
    # Timing comparison
    print("\n" + "=" * 50)
    print("TIMING SUMMARY")
    print("=" * 50)
    print(f"NumPy time:              {t_np:.3f} s")
    print(f"JAX time (with JIT):     {t_jax_first:.3f} s")
    print(f"JAX time (compiled):     {t_jax_compiled:.3f} s")
    if t_jax_compiled > 0:
        speedup = t_np / t_jax_compiled
        print(f"Speedup (compiled JAX):  {speedup:.1f}x")
    
    all_pass = counts_match and cors_match
    print("\n" + "=" * 50)
    if all_pass:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("=" * 50)
    
    return all_pass


def run_quick_test():
    """Run a quick test with mu_bin1=0 and mu_bin2=0 (smallest bins, fastest)."""
    return test_jax_correlations(mu_bin1=0, mu_bin2=0)


if __name__ == "__main__":
    import sys
    
    # Check for JAX
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print()
    except ImportError:
        print("JAX is not installed. Install with: pip install jax jaxlib")
        print("For GPU support: pip install jax[cuda12]")
        sys.exit(1)
    
    # Run the test
    success = run_quick_test()
    sys.exit(0 if success else 1)
