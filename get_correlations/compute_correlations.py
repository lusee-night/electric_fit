#!/usr/bin/env python
"""
Driver script to compute correlations and save to numpy file.

Usage:
    python compute_correlations.py [OPTIONS]

Options:
    --jax / --no-jax     Use JAX-accelerated version (default: no-jax)
    --gpu / --cpu        Use GPU or CPU (default: cpu, only applies with --jax)
    --batch-size SIZE    Batch size for JAX version (default: 1000000)
    --maxpix N           Maximum number of pixels to process (default: all)
    --output FILE        Output file path (default: correlations.npz)
    --map PATH           Path to FITS map file

Examples:
    python compute_correlations.py --jax --gpu --batch-size 500000
    python compute_correlations.py --no-jax --maxpix 1000
    python compute_correlations.py --jax --cpu --output my_correlations.npz
"""
import argparse
import os
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(
        description='Compute correlations from HEALPix map.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # JAX vs non-JAX
    jax_group = parser.add_mutually_exclusive_group()
    jax_group.add_argument('--jax', action='store_true', dest='use_jax',
                           help='Use JAX-accelerated version')
    jax_group.add_argument('--no-jax', action='store_false', dest='use_jax',
                           help='Use standard NumPy version (default)')
    parser.set_defaults(use_jax=False)
    
    # GPU vs CPU
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--gpu', action='store_const', const='cuda', dest='device',
                              help='Use GPU (CUDA) for JAX computation')
    device_group.add_argument('--cpu', action='store_const', const='cpu', dest='device',
                              help='Use CPU for JAX computation (default)')
    parser.set_defaults(device='cpu')
    
    # Other options
    parser.add_argument('--batch-size', type=int, default=1000000,
                        help='Batch size for JAX version (default: 1000000)')
    parser.add_argument('--maxpix', type=int, default=None,
                        help='Maximum number of pixels to process (default: all)')
    parser.add_argument('--output', '-o', type=str, default='correlations.npz',
                        help='Output file path (default: correlations.npz)')
    parser.add_argument('--map', type=str, 
                        default='../../Drive/Simulations/SkyModels/ULSA_maps/200.fits',
                        help='Path to FITS map file')
    
    args = parser.parse_args()
    
    # Set JAX platform before importing Correlator (which imports JAX)
    if args.use_jax:
        os.environ['JAX_PLATFORMS'] = args.device
        print(f"Using JAX with device: {args.device}")
    
    # Import Correlator after setting JAX platform
    from Correlator import Correlator
    
    print(f"Loading map from: {args.map}")
    c = Correlator(args.map)
    
    start_time = time.time()
    
    if args.use_jax:
        print(f"Computing correlations with JAX (batch_size={args.batch_size}, maxpix={args.maxpix})")
        cor, count = c.get_correlations_jax(maxpix=args.maxpix, batch_size=args.batch_size)
    else:
        print(f"Computing correlations with NumPy (maxpix={args.maxpix})")
        cor, count = c.get_correlations(maxpix=args.maxpix)
    
    elapsed = time.time() - start_time
    print(f"Computation completed in {elapsed:.2f} seconds")
    
    np.savez(args.output, cor=cor, count=count)
    print(f"Correlations saved to {args.output}")


if __name__ == "__main__":
    exit(main())
