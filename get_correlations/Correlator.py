#
# Class to load healpix maps and computer spatially resolved correlations
#
import healpy as hp
import numpy as np
import fitsio
import tqdm

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class Correlator:

    mu_bins = [0.0, 0.05, 0.2, 0.5, 1.0]
    

    def __init__ (self, map_path, mu_bin1, mu_bin2, dDist = 0.1):
        self.map = fitsio.read(map_path)
        self.nside = hp.get_nside(self.map[0,:])
        self.xyz = np.array(hp.pix2vec(self.nside, np.arange(hp.nside2npix(self.nside)))).T
        zs = np.abs(self.xyz[:, 2])
        self.bin1_ndx = np.where((zs>=self.mu_bins[mu_bin1])& (zs<=self.mu_bins[mu_bin1+1]))[0]
        self.bin2_ndx = np.where((zs>=self.mu_bins[mu_bin2])& (zs<=self.mu_bins[mu_bin2+1]))[0]
        self.dDist = dDist
        self.nD = int(np.pi//dDist) + 1
        self.nFreq= self.map.shape[0]

    def get_correlations(self):
        self.map1_pixels = self.map[:,self.bin1_ndx]
        self.map2_pixels = self.map[:,self.bin2_ndx]
        self.N1 = self.map1_pixels.shape[1]
        self.N2 = self.map2_pixels.shape[1]
        self.map1_pos = self.xyz[self.bin1_ndx,:]
        self.map2_pos = self.xyz[self.bin2_ndx,:]

        cors = np.zeros((self.nD, self.nFreq, self.nFreq))
        counts = np.zeros(self.nD)

        for i in tqdm.tqdm(range(self.N1)):
            dist = np.dot(self.map1_pos[i,:], self.map2_pos.T)
            dist[dist>1.0] = 1.0
            dist[dist<-1.0] = -1.0
            dist = np.arccos(dist)
            try:
                assert (np.all(dist>=0) and np.all(dist<=np.pi))
            except:
                print(dist.min(), dist.max())
                raise
            bins = (dist//self.dDist).astype(int)
            assert (np.all(bins>=0) and np.all(bins<self.nD))
            counts += np.bincount(bins, minlength=self.nD)
            for j in range(self.N2):
                d = bins[j]
                cors[d,:,:] += np.outer(self.map1_pixels[:,i], self.map2_pixels[:,j])

        for d in range(self.nD):
            if counts[d]>0:
                cors[d,:,:] /= counts[d]
        return cors, counts

    def get_correlations_jax(self, batch_size=128):
        """
        JAX-accelerated version of get_correlations.
        Uses vectorized operations and JIT compilation for GPU acceleration.
        Processes in batches to avoid GPU memory issues.
        
        Parameters
        ----------
        batch_size : int
            Number of pixels from bin1 to process at once. Smaller values use less
            GPU memory but may be slower. Default is 128.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Install it with: pip install jax jaxlib")
        
        # Enable float64 for accurate binning (must be set before JIT compilation)
        jax.config.update("jax_enable_x64", True)
        
        map1_pixels = self.map[:, self.bin1_ndx]  # (nFreq, N1)
        map2_pixels = self.map[:, self.bin2_ndx]  # (nFreq, N2)
        map1_pos = self.xyz[self.bin1_ndx, :]  # (N1, 3)
        map2_pos = self.xyz[self.bin2_ndx, :]  # (N2, 3)
        N1 = map1_pixels.shape[1]

        # Convert to JAX arrays - use float64 for positions to match NumPy precision in binning
        map1_pixels_jax = jnp.array(map1_pixels, dtype=jnp.float32)
        map2_pixels_jax = jnp.array(map2_pixels, dtype=jnp.float32)
        map1_pos_jax = jnp.array(map1_pos, dtype=jnp.float64)
        map2_pos_jax = jnp.array(map2_pos, dtype=jnp.float64)

        nD = self.nD
        nFreq = self.nFreq
        dDist = self.dDist

        @jit
        def compute_batch_correlations(map1_pixels_batch, map1_pos_batch, map2_pixels_jax, map2_pos_jax):
            """Process a batch of pixels from bin1 against all pixels from bin2."""
            # Compute pairwise distances for this batch: (batch_size, N2)
            dot_products = jnp.dot(map1_pos_batch, map2_pos_jax.T)
            dot_products = jnp.clip(dot_products, -1.0, 1.0)
            distances = jnp.arccos(dot_products)
            
            # Compute bin indices for all pairs (use floor division like NumPy)
            bins = jnp.floor(distances / dDist).astype(jnp.int32)
            bins = jnp.clip(bins, 0, nD - 1)

            # Compute outer products for all frequency pairs using einsum
            # map1_pixels_batch: (nFreq, batch_size), map2_pixels_jax: (nFreq, N2)
            # Result shape: (batch_size, N2, nFreq, nFreq)
            outer_products = jnp.einsum('fi,gj->ijfg', map1_pixels_batch, map2_pixels_jax)

            # Flatten for segment operations
            flat_bins = bins.flatten()  # (batch_size * N2,)
            flat_outers = outer_products.reshape(-1, nFreq, nFreq)  # (batch_size * N2, nFreq, nFreq)

            # Sort by bin index for deterministic segment_sum
            sort_indices = jnp.argsort(flat_bins)
            sorted_bins = flat_bins[sort_indices]
            sorted_outers = flat_outers[sort_indices]

            # Use segment_sum for deterministic accumulation (works correctly on GPU)
            cors_batch = jax.ops.segment_sum(sorted_outers, sorted_bins, num_segments=nD)

            # Count pairs per bin using segment_sum
            ones = jnp.ones(flat_bins.shape[0], dtype=jnp.float32)
            sorted_ones = ones[sort_indices]
            counts_batch = jax.ops.segment_sum(sorted_ones, sorted_bins, num_segments=nD)

            return cors_batch, counts_batch

        # Process in batches and accumulate results
        cors_total = np.zeros((nD, nFreq, nFreq), dtype=np.float32)
        counts_total = np.zeros(nD, dtype=np.float32)

        for start in tqdm.tqdm(range(0, N1, batch_size)):
            end = min(start + batch_size, N1)
            
            # Get batch
            map1_pixels_batch = map1_pixels_jax[:, start:end]
            map1_pos_batch = map1_pos_jax[start:end, :]
            
            # Compute correlations for this batch
            cors_batch, counts_batch = compute_batch_correlations(
                map1_pixels_batch, map1_pos_batch, map2_pixels_jax, map2_pos_jax
            )
            
            # Accumulate
            cors_total += np.array(cors_batch)
            counts_total += np.array(counts_batch)

        # Normalize
        for d in range(nD):
            if counts_total[d] > 0:
                cors_total[d] /= counts_total[d]

        return cors_total, counts_total