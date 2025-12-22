#
# Class to load healpix maps and computer spatially resolved correlations
#
import healpy as hp
import numpy as np
import fitsio
import tqdm

import os
# Set JAX platform before importing. Default to CPU for compatibility.
# To use GPU, set environment variable JAX_PLATFORMS='cuda' before running.
if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class Correlator:


    def __init__ (self, map_path, el_param=500, Ndist=32, Ng = 20):
        self.map = fitsio.read(map_path)
        x=np.arange(1,51)
        renorm = 4e5*(x/10)**-2.5
        self.map = self.map / renorm[:,None]
        self.npix = self.map.shape[1]
        self.nside = hp.get_nside(self.map[0,:])
        self.xyz = np.array(hp.pix2vec(self.nside, np.arange(hp.nside2npix(self.nside)))).T
        zs = np.arcsin(self.xyz[:,2])
        xs = np.arctan2(self.xyz[:,1], self.xyz[:,0])
        self.Nd = Ndist
        self.Ng = Ng
        self.el_param = el_param

        self.elmap = np.sqrt((zs**2+xs**2/self.el_param))

        self.gposmax = self.elmap.max()
        print (self.gposmax,'XX')
        self.elmap /= self.gposmax
        self.gbinmap = np.floor(self.elmap*self.Ng).astype(int)
        self.gbinmap[self.gbinmap>=self.Ng] = self.Ng-1

        self.dDist = (np.pi+1e-3) / self.Nd
        self.nFreq= self.map.shape[0]
        
    

    def get_correlations(self,maxpix = None):
        
        cors = np.zeros((self.Ng, self.Nd, self.nFreq, self.nFreq))
        counts = np.zeros((self.Ng, self.Nd))

        for i in tqdm.tqdm(range(self.npix if maxpix is None else maxpix)):
            dist = np.dot(self.xyz[i,:], self.xyz[i:,:].T)
            dist[dist>1.0] = 1.0
            dist[dist<-1.0] = -1.0
            dist = np.arccos(dist)
            dbins = (dist//self.dDist).astype(int)
            gposv = self.xyz[i,:]+self.xyz[i:,:]
            norm =  np.linalg.norm(gposv, axis=1)
            gposv /= norm[:,None]
            zs = np.arcsin(gposv[:,2])
            xs = np.arctan2(gposv[:,1], gposv[:,0])
            gpos = np.sqrt((zs**2+xs**2/self.el_param))/self.gposmax
            gbins = (gpos*self.Ng).astype(int)
            gbins[gbins>=self.Ng] = self.Ng-1
            gbins [norm<1e-6] = self.Ng-1  # handle zero vector case
            #if (bins.min()<4):
            #    print (dist.min(), dist.max(), bins.min(), bins.max())
            assert (np.all(dbins>=0) and np.all(dbins<self.Nd))
            for j,gbin,dbin in zip(range(i,self.npix), gbins, dbins):                
                cors[gbin, dbin ,:,:] += np.outer(self.map[:,i], self.map[:,j])                
                counts[gbin, dbin] += 1
                if (gbin==0 and dbin==0):
                    print (gbin,dbin)
                    print(cors[gbin,dbin])
    
        counts[counts==0] = 1  # avoid division by zero
        
        cors /= counts[:, :,  None, None]
        return cors, counts

    def get_correlations_jax(self, maxpix=None, batch_size=None):
        """
        JAX-accelerated version of get_correlations.
        Uses the natural i-loop structure where each iteration processes all j >= i.
        
        Parameters:
        -----------
        maxpix : int, optional
            Maximum number of pixels to process for i loop. If None, process all.
        batch_size : int, optional
            Ignored (kept for API compatibility). Natural batching from i-loop is used.
            
        Returns:
        --------
        cors : ndarray
            Correlation array of shape (Ng, Nd, nFreq, nFreq)
        counts : ndarray
            Count array of shape (Ng, Nd)
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available. Please install jax and jaxlib.")
        
        npix_i = self.npix if maxpix is None else maxpix
        npix_total = self.npix
        
        # Convert data to JAX arrays
        xyz_jax = jnp.array(self.xyz, dtype=jnp.float32)
        map_jax = jnp.array(self.map, dtype=jnp.float32)
        
        # Constants
        Ng = self.Ng
        Nd = self.Nd
        nFreq = self.nFreq
        dDist = float(self.dDist)
        el_param = float(self.el_param)
        gposmax = float(self.gposmax)
        flat_size = Ng * Nd
        
        @jit
        def process_pixel_i(i, xyz, sky_map, cors_acc, counts_acc):
            """
            Process all pairs (i, j) for j >= i and accumulate directly on GPU.
            
            Returns updated cors and counts accumulators.
            """
            xyz_i = xyz[i]  # (3,)
            map_i = sky_map[:, i]  # (nFreq,)
            
            # Create mask for valid j >= i
            j_indices = jnp.arange(xyz.shape[0])
            valid_mask = j_indices >= i
            
            # Compute for ALL j
            # Angular distance
            dot_prod = jnp.dot(xyz, xyz_i)  # (npix,)
            dot_prod = jnp.clip(dot_prod, -1.0, 1.0)
            dist = jnp.arccos(dot_prod)
            dbins = (dist // dDist).astype(jnp.int32)
            dbins = jnp.clip(dbins, 0, Nd - 1)
            
            # Geometric position
            gposv = xyz_i + xyz  # (npix, 3)
            norm = jnp.linalg.norm(gposv, axis=1)
            safe_norm = jnp.where(norm < 1e-6, 1.0, norm)
            gposv_normalized = gposv / safe_norm[:, None]
            
            gpos = jnp.sqrt(jnp.arcsin(gposv_normalized[:, 2])**2 + jnp.arctan2(gposv_normalized[:, 1], gposv_normalized[:, 0])**2 / el_param) / gposmax
            gbins = (gpos * Ng).astype(jnp.int32)
            gbins = jnp.clip(gbins, 0, Ng - 1)
            gbins = jnp.where(norm < 1e-6, Ng - 1, gbins)
            
            # Flat bin index - use out-of-bounds index for invalid entries
            flat_bins = gbins * Nd + dbins
            # Set invalid bins to flat_size (will be ignored in segment_sum)
            flat_bins = jnp.where(valid_mask, flat_bins, flat_size)
            
            # Compute outer products only for valid j and accumulate per bin
            # outer_prods[k, f, g] = map_i[f] * sky_map[g, k]
            outer_prods = map_i[:, None] * sky_map  # (nFreq, npix)
            
            # For each bin, sum the outer products
            # We need to accumulate: for each flat_bin b, sum over k where flat_bins[k]==b
            # cors_acc[b, f, g] += sum over k: outer_prods[f, k] * sky_map[g, k] where flat_bins[k] == b
            
            # Use segment_sum for counts
            ones = jnp.where(valid_mask, 1, 0)
            count_contrib = jax.ops.segment_sum(ones, flat_bins, num_segments=flat_size + 1)[:flat_size]
            counts_acc = counts_acc + count_contrib
            
            # For correlations, we need to be smarter
            # cors[b, f, g] = sum over k in bin b: map_i[f] * sky_map[g, k]
            # = map_i[f] * (sum over k in bin b: sky_map[g, k])
            # So for each bin b and freq g, compute: sum over k in bin b: sky_map[g, k]
            
            # map_sum_per_bin[b, g] = sum over k where flat_bins[k]==b: sky_map[g, k]
            # Use segment_sum for each frequency
            def sum_freq_g(g_map):
                # g_map: (npix,) - sky_map[g, :]
                masked = jnp.where(valid_mask, g_map, 0.0)
                return jax.ops.segment_sum(masked, flat_bins, num_segments=flat_size + 1)[:flat_size]
            
            map_sum_per_bin = jax.vmap(sum_freq_g)(sky_map)  # (nFreq, flat_size)
            
            # cors_contrib[b, f, g] = map_i[f] * map_sum_per_bin[g, b]
            # = map_i[f] * map_sum_per_bin.T[b, g]
            cors_contrib = map_i[:, None, None] * map_sum_per_bin.T[None, :, :]  # (nFreq, flat_size, nFreq)
            # Need shape (flat_size, nFreq, nFreq), so transpose
            cors_contrib = jnp.transpose(cors_contrib, (1, 0, 2))  # (flat_size, nFreq, nFreq)
            
            cors_acc = cors_acc + cors_contrib
            
            return cors_acc, counts_acc
        
        # Initialize accumulators on GPU
        cors_acc = jnp.zeros((flat_size, nFreq, nFreq), dtype=jnp.float32)
        counts_acc = jnp.zeros(flat_size, dtype=jnp.int32)
        
        # Warm up JIT
        print("Warming up JIT compilation...")
        cors_acc, counts_acc = process_pixel_i(0, xyz_jax, map_jax, cors_acc, counts_acc)
        jax.block_until_ready(cors_acc)
        # Reset after warmup
        cors_acc = jnp.zeros((flat_size, nFreq, nFreq), dtype=jnp.float32)
        counts_acc = jnp.zeros(flat_size, dtype=jnp.int32)
        print("JIT compilation complete.")
        
        # Main loop over i
        for i in tqdm.tqdm(range(npix_i)):
            cors_acc, counts_acc = process_pixel_i(i, xyz_jax, map_jax, cors_acc, counts_acc)
        
        # Transfer to CPU
        cors = np.array(cors_acc).reshape(Ng, Nd, nFreq, nFreq).astype(np.float64)
        counts = np.array(counts_acc).reshape(Ng, Nd).astype(np.int64)
        
        # Normalize
        counts_safe = np.where(counts == 0, 1, counts)
        cors /= counts_safe[:, :, None, None]
        
        return cors, counts