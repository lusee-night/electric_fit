#!/usr/bin/env python
"""
Driver script to compute correlations using JAX/GPU and save to numpy file.

Usage:
    python compute_correlations.py <mu_bin1> <mu_bin2> [--output OUTPUT] [--map MAP_PATH] [--batch-size BATCH_SIZE]

Examples:
    python compute_correlations.py 0 0
    python compute_correlations.py 1 2 --output my_correlations.npz
    python compute_correlations.py 0 0 --batch-size 64
"""
import argparse
import numpy as np
import time
from Correlator import Correlator


def compute_and_save_correlations(mu_bin1, mu_bin2, map_path, output_path, batch_size=128):
    """
    Compute correlations using JAX/GPU and save to numpy file.
    
    Parameters
    ----------
    mu_bin1 : int
        First mu bin index (0-3)
    mu_bin2 : int
        Second mu bin index (0-3)
    map_path : str
        Path to the FITS map file
    output_path : str
        Output path for the numpy file
    batch_size : int
        Batch size for JAX processing
    
    Returns
    -------
    tuple
        (correlations, counts) arrays
    """
    print(f"Computing correlations for mu_bin1={mu_bin1}, mu_bin2={mu_bin2}")
    print(f"Map: {map_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    # Create correlator
    corr = Correlator(map_path, mu_bin1, mu_bin2)
    print(f"N1 (bin1 pixels): {len(corr.bin1_ndx)}")
    print(f"N2 (bin2 pixels): {len(corr.bin2_ndx)}")
    print(f"nFreq: {corr.nFreq}")
    print(f"nD (distance bins): {corr.nD}")
    
    # Compute correlations using JAX
    print("\nComputing correlations with JAX...")
    t0 = time.time()
    cors, counts = corr.get_correlations_jax(batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"Computation time: {elapsed:.3f} seconds")
    
    # Save to numpy file
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, 
             correlations=cors, 
             counts=counts,
             mu_bin1=mu_bin1,
             mu_bin2=mu_bin2,
             nFreq=corr.nFreq,
             nD=corr.nD,
             dDist=corr.dDist,
             mu_bins=corr.mu_bins)
    print("Done!")
    
    return cors, counts


def main():
    parser = argparse.ArgumentParser(
        description="Compute correlations using JAX/GPU and save to numpy file."
    )
    parser.add_argument("mu_bin1", type=int, help="First mu bin index (0-3)")
    parser.add_argument("mu_bin2", type=int, help="Second mu bin index (0-3)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: correlations_<mu_bin1>_<mu_bin2>.npz)")
    parser.add_argument("--map", "-m", type=str, 
                        default="../../Drive/Simulations/SkyModels/ULSA_maps/200.fits",
                        help="Path to the FITS map file")
    parser.add_argument("--batch-size", "-b", type=int, default=128,
                        help="Batch size for JAX processing (default: 128)")
    
    args = parser.parse_args()
    
    # Validate bin indices
    if not (0 <= args.mu_bin1 <= 3):
        parser.error(f"mu_bin1 must be 0-3, got {args.mu_bin1}")
    if not (0 <= args.mu_bin2 <= 3):
        parser.error(f"mu_bin2 must be 0-3, got {args.mu_bin2}")
    
    # Set default output path if not provided
    output_path = args.output
    if output_path is None:
        output_path = f"correlations_{args.mu_bin1}_{args.mu_bin2}.npz"
    
    # Check for JAX
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print()
    except ImportError:
        print("ERROR: JAX is not installed. Install with: pip install jax jaxlib")
        print("For GPU support: pip install jax[cuda12]")
        return 1
    
    # Compute and save
    compute_and_save_correlations(
        args.mu_bin1, 
        args.mu_bin2, 
        args.map, 
        output_path,
        args.batch_size
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
