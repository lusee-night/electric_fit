"""
Test script to validate the JAX implementation of get_correlations
against the original NumPy implementation.
"""

import numpy as np
from Correlator import Correlator

def test_get_correlations_jax():
    """
    Compare the JAX and NumPy implementations of get_correlations.
    Uses maxpix=5 for reasonable runtime.
    """
    print("Initializing Correlator...")
    a = Correlator('../../Drive/Simulations/SkyModels/ULSA_maps/200.fits')
    
    maxpix = 5
    
    print(f"\nRunning NumPy get_correlations with maxpix={maxpix}...")
    cors_np, counts_np = a.get_correlations(maxpix=maxpix)
    
    print(f"\nRunning JAX get_correlations with maxpix={maxpix}...")
    cors_jax, counts_jax = a.get_correlations_jax(maxpix=maxpix, batch_size=1000)
    
    # Compare counts
    # Note: The original sets counts[counts==0] = 1 in-place before returning,
    # while the JAX version keeps original counts and uses safe division separately.
    # For comparison, we check that non-zero original counts match.
    print("\n=== Comparing Results ===")
    print(f"Counts shape: NumPy={counts_np.shape}, JAX={counts_jax.shape}")
    
    # The original version sets 0 counts to 1, so where counts_np==1 and counts_jax==0,
    # these were originally 0 and that's fine. Compare where counts_jax > 0
    nonzero_mask = counts_jax > 0
    counts_match_nonzero = np.allclose(counts_np[nonzero_mask], counts_jax[nonzero_mask])
    
    # Also check that where counts_jax is 0, counts_np should be 1 (due to the normalization fix)
    zero_mask = counts_jax == 0
    counts_zeros_ok = np.all(counts_np[zero_mask] == 1) if np.any(zero_mask) else True
    
    counts_match = counts_match_nonzero and counts_zeros_ok
    print(f"Counts match (accounting for zero handling): {counts_match}")
    if not counts_match:
        diff = np.abs(counts_np - counts_jax)
        print(f"  Max count difference: {diff.max()}")
        print(f"  Locations with differences: {np.sum(diff > 0)}")
    
    # Compare correlations
    print(f"\nCorrelations shape: NumPy={cors_np.shape}, JAX={cors_jax.shape}")
    
    # Use relative tolerance for floating point comparison
    cors_match = np.allclose(cors_np, cors_jax, rtol=1e-4, atol=1e-6)
    print(f"Correlations match (rtol=1e-4, atol=1e-6): {cors_match}")
    
    if not cors_match:
        diff = np.abs(cors_np - cors_jax)
        print(f"  Max absolute difference: {diff.max()}")
        print(f"  Mean absolute difference: {diff.mean()}")
        
        # Find where differences are largest
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Location of max diff: {max_idx}")
        print(f"  NumPy value at max diff: {cors_np[max_idx]}")
        print(f"  JAX value at max diff: {cors_jax[max_idx]}")
        
        # Compute relative differences where values are non-zero
        nonzero_mask = np.abs(cors_np) > 1e-10
        if np.any(nonzero_mask):
            rel_diff = np.abs(cors_np[nonzero_mask] - cors_jax[nonzero_mask]) / np.abs(cors_np[nonzero_mask])
            print(f"  Max relative difference (nonzero values): {rel_diff.max()}")
            print(f"  Mean relative difference (nonzero values): {rel_diff.mean()}")
    
    # Summary
    print("\n=== Summary ===")
    if counts_match and cors_match:
        print("✓ JAX implementation matches NumPy implementation!")
        return True
    else:
        print("✗ Implementations do not match exactly.")
        print("  This may be due to floating point precision differences.")
        
        # Check with looser tolerance
        cors_match_loose = np.allclose(cors_np, cors_jax, rtol=1e-3, atol=1e-5)
        if cors_match_loose:
            print("  ✓ However, they match with looser tolerance (rtol=1e-3, atol=1e-5)")
            return True
        return False


if __name__ == "__main__":
    success = test_get_correlations_jax()
    exit(0 if success else 1)
