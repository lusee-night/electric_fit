# JAX matrix-free GLS for m-hat and its covariance

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jax.scipy.linalg import cho_solve


def build_adjoint(fun, out_shape_dtype):
    # Given y = fun(x), return a function that applies fun^T to a vector
    lt = jax.linear_transpose(fun, out_shape_dtype)  # primals spec
    # lt(ct) returns a tuple; take first element which is cotangent for the primal input
    return lambda v: lt(v)[0]

def solve_m_and_cov(d, Ns, Nd, Nf, Ps, S_op, Dm, N_diag, tol=1e-6, maxiter=500, precond=None):
    """Iterative (matrix-free) GLS solution for m-hat and its covariance.

    Parameters
    ----------
    d : (Nd,) data vector
    Ns, Nd, Nf : ints (problem sizes)
    Ps : function s->P s
    S_op : function v->S v
    Dm : function m->Dcal m
    tol : CG relative tolerance
    maxiter : maximum CG iterations
    precond : optional function v -> M^{-1} v (left preconditioner)
    jitter : float, optional diagonal added to D = P S P^T + jitter I (must
             match any jitter used in the dense reference for apples-to-apples)
    """
    # Adjoint ops from the forward definitions
    # We only need shapes+dtypes to create the transposes
    s_spec  = jax.ShapeDtypeStruct((Ns,), d.dtype)
    m_spec  = jax.ShapeDtypeStruct((Nf,), d.dtype)
    PsT     = build_adjoint(Ps, s_spec)    # P^T
    DmT     = build_adjoint(Dm, m_spec)    # D^T

    # D matvec: R^{Nd} -> R^{Nd}, v |-> P S P^T v
    def D_matvec(v):
        return Ps(S_op(PsT(v))) + N_diag * v

    # Optional left preconditioner for CG: M^{-1} ~ D^{-1}
    # Provide function M(v) ≈ D^{-1} v (Jacobi or block-Jacobi; see notes below).
    M = precond

    # Solve Dx = rhs with CG (matrix-free)
    def solve_D(rhs):

        x, info = cg(D_matvec, rhs, tol=tol, maxiter=maxiter, M=M)
        
        # info: 0 -> converged, >0 -> iter count (not converged), <0 -> breakdown
        if info is not None:
            # info can be a JAX array; convert to host int when possible
            try:
                info_val = int(info)
            except Exception:
                info_val = info
            if info_val != 0:
                print(f"WARNING: CG did not converge (info={info_val})")
        # Basic NaN / Inf guard (helps diagnose issues early)
        #if jnp.isnan(x).any() or jnp.isinf(x).any():
        #    raise FloatingPointError("CG produced NaNs/Infs; check SPD-ness of D and preconditioner.")
        return x

    # b = D^T D^{-1} d = Dm^T (D^{-1} d)
    
    x_d = solve_D(d)
    b   = DmT(x_d)
    #print ("b:", b[:10])
    # Build A = Dm^T D^{-1} Dm (Nf x Nf) using Nf solves
    I_nf   = jnp.eye(Nf, dtype=d.dtype)               # columns are e_j
    cols_Dm = jax.vmap(lambda e: Dm(e), in_axes=1, out_axes=1)(I_nf)  # Nd x Nf

    # Solve D X = (Dm * I), column by column in parallel
    solve_col = jax.vmap(solve_D, in_axes=1, out_axes=1)             # Nd x Nf -> Nd x Nf
    X = solve_col(cols_Dm)                                           # Nd x Nf

    

    # A columns are Dm^T X[:, j]
    col_to_Atcol = lambda x_col: DmT(x_col)                          # Nd -> Nf
    A = jax.vmap(col_to_Atcol, in_axes=1, out_axes=1)(X)             # Nf x Nf
    A = 0.5 * (A + A.T)                                              # symmetrize numerically

    #print ("A:", A[:10,:10])
    # Solve for m-hat and covariance (small Nf x Nf system)
    # If A is not numerically SPD (can happen from CG noise), add minimal jitter
    try:
        L = jnp.linalg.cholesky(A)
    except Exception:
        eps = 1e-10 * jnp.trace(A)/A.shape[0]
        L = jnp.linalg.cholesky(A + eps * jnp.eye(Nf, dtype=A.dtype))

    m_hat = cho_solve((L, True), b)                     # A m = b
    #print ("m_hat:", m_hat[:10])
    C     = cho_solve((L, True), jnp.eye(Nf, dtype=d.dtype))
    return m_hat, C



def materialize_small_mats(Ps, S_op, Dm, Ns, Nd, Nf):
    """
    Build dense P, S, Dcal by pushing standard basis vectors through your ops.
    WARNING: use only for tiny problems.
    """
    I_Ns = jnp.eye(Ns, dtype=jnp.float64)
    I_Nf = jnp.eye(Nf, dtype=jnp.float64)

    # Columns: P[:,i] = Ps(e_i), S[:,i] = S_op(e_i), Dcal[:,j] = Dm(e_j)
    P     = jax.vmap(Ps,   in_axes=1, out_axes=1)(I_Ns)    # (Nd, Ns)
    S     = jax.vmap(S_op, in_axes=1, out_axes=1)(I_Ns)    # (Ns, Ns)
    Dcal  = jax.vmap(Dm,   in_axes=1, out_axes=1)(I_Nf)    # (Nd, Nf)
    return P, S, Dcal



def bruteforce_gls(d, P, S, Dcal, N_diag):
    """
    Dense reference implementation (for small problems).

    Args
    ----
    d     : (Nd,)         data vector
    P     : (Nd, Ns)      forward operator 𝒫
    S     : (Ns, Ns)      sky covariance S
    Dcal  : (Nd, Nf)      design matrix 𝒟
    jitter: float         optional diagonal jitter added to D

    Returns
    -------
    m_hat : (Nf,)         GLS estimator
    C     : (Nf, Nf)      estimator covariance
    """
    # D = P S P^T
    D = P @ S @ P.T + jnp.diag(N_diag)

    # Build A = Dcal^T D^{-1} Dcal and b = Dcal^T D^{-1} d via Cholesky solves
    Ld = jnp.linalg.cholesky(D)                             # D = Ld Ld^T
    X  = cho_solve((Ld, True), Dcal)                        # solves D X = Dcal
    b  = Dcal.T @ cho_solve((Ld, True), d)                  # b = Dcal^T D^{-1} d
    A  = Dcal.T @ X                                         # A = Dcal^T D^{-1} Dcal
    A  = 0.5 * (A + A.T)                                    # symmetrize numerically

    # Solve A m = b and form C = A^{-1} (small Nf×Nf)
    La   = jnp.linalg.cholesky(A)
    m_hat = cho_solve((La, True), b)
    C     = cho_solve((La, True), jnp.eye(A.shape[0], dtype=A.dtype))
    return m_hat, C
