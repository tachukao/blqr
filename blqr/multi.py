"""Batched finite-time linear quadratic regulator (batch_lqr)

Adapted from https://github.com/google/trajax/blob/main/trajax/tvlqr.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp
import jax.scipy as sp


@jax.jit
def batch_uncontrolled_rollout(x0, A, B, d):
    """Batched uncontrolled roll-out time-varying dynamics x[t+1] = A[t] x[t] + d[t]."""
    T, batch_size, n, m = B.shape
    X = jnp.zeros((T + 1, batch_size, n))
    X = jax.ops.index_update(X, jax.ops.index[0], x0)

    def body(t, X):
        x = jax.vmap(jnp.matmul)(A[t], X[t]) + d[t]
        X = jax.ops.index_update(X, jax.ops.index[t + 1], x)
        return X

    return jax.lax.fori_loop(0, T, body, X)


@jax.jit
def batch_lqr_step(V, v, Q, q, R, r, M, A, B, d, delta=1e-8):
    """Single Batch LQR Step.
    Args:
      V: [batch_size, n, batch_size, n] numpy array.
      v: [batch_size, n] numpy array.
      Q: [batch_size, n, n] numpy array.
      q: [batch_size, n] numpy array.
      R: [m, m] numpy array.
      r: [m] numpy array.
      M: [batch_size, n, m] numpy array.
      A: [batch_size, n, n] numpy array.
      B: [batch_size, n, m] numpy array.
      d: [batch_size, n] numpy array.
      delta: Enforces positive definiteness by ensuring smallest eigenval > delta.
    Returns:
      V, v: updated matrices encoding quadratic value function.
      K, k: state feedback gain and affine term.
    """
    batch_size, n, m = B.shape
    symmetrize = lambda x: (x + x.T) / 2
    symmetrize_full = lambda x: (x + x.transpose(2, 3, 0, 1)) / 2

    AtV = jnp.einsum("...ji,...jkl", A, V)
    AtVA = symmetrize_full(jnp.einsum("ai...j,...jk->ai...k", AtV, A))
    BtV = jnp.einsum("ijk,ijlm", B, V)  # (m, batch_size, n)
    BtVA = jnp.einsum("i...k,...km->...im", BtV, A)
    BtVB = jnp.einsum("ijk,jkl", BtV, B)
    G = symmetrize(R + jnp.einsum("ijk,jkl", BtV, B))
    # make G positive definite so that smallest eigenvalue > delta.
    S, _ = jnp.linalg.eigh(G)
    G_ = G + jnp.maximum(0.0, delta - S[0]) * jnp.eye(G.shape[0])

    H = BtVA + M.transpose(0, 2, 1)  # (batch_size, m, n)
    h = jnp.einsum("ijk,ij", B, v) + jnp.einsum("ijk,jk", BtV, d) + r

    vlinsolve = jax.vmap(
        lambda x, y: sp.linalg.solve(x, y, sym_pos=True), in_axes=(None, 0)
    )
    K = -vlinsolve(G_, H)  # (batch_size, m, n)
    k = -sp.linalg.solve(G_, h, sym_pos=True)  # (m, )

    H_GK = H + jnp.einsum("ij,ajk->aik", G, K)  # (batch_size, m, n)
    V = symmetrize_full(
        sp.linalg.block_diag(*Q).reshape((batch_size, n, batch_size, n))
        + AtVA
        + jnp.einsum("ijk,mjo->ikmo", H_GK, K)
        + jnp.einsum("ijk,mjo->ikmo", K, H)
    )

    v = (
        q
        + jax.vmap(jnp.matmul)(A.transpose(0, 2, 1), v)
        + jnp.einsum("ijkl,kl", AtV, d)
        + jnp.matmul(H_GK.transpose(0, 2, 1), k)
        + jnp.matmul(K.transpose(0, 2, 1), h)
    )

    return V, v, K, k


@jax.jit
def batch_rollout(K, k, x0, A, B, d):
    """Batched rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""

    T, batch_size, m, n = K.shape
    X = jnp.zeros((T + 1, batch_size, n))
    U = jnp.zeros((T, m))
    X = jax.ops.index_update(X, jax.ops.index[0], x0)

    def body(t, inputs):
        X, U = inputs
        u = jnp.einsum("ijk,ik", K[t], X[t]) + k[t]
        x = jax.vmap(jnp.matmul)(A[t], X[t]) + jnp.matmul(B[t], u) + d[t]
        X = jax.ops.index_update(X, jax.ops.index[t + 1], x)
        U = jax.ops.index_update(U, jax.ops.index[t], u)
        return X, U

    return jax.lax.fori_loop(0, T, body, (X, U))


@jax.jit
def batch_lqr(Qf, qf, Q, q, R, r, M, A, B, d):
    """Batched discrete-time Finite Horizon Time-varying LQR.
    Args:
      Qf: [batch_size, n, n] numpy array.
      qf: [batch_size, n] numpy array.
       Q: [T, batch_size, n, n] numpy array.
       q: [T, batch_size, n] numpy array.
       R: [T, m, m] numpy array.
       r: [T, m] numpy array.
       M: [T, batch_size, n, m] numpy array.
       A: [T, batch_size, n, n] numpy array.
       B: [T, batch_size, n, m] numpy array.
       d: [T, batch_size, n] numpy array.
    Returns:
       K: [T, m, n] Gains
       k: [T, m] Affine terms (u_t = jnp.matmul(K[t],  x_t) + k[t])
       V: [batch, n, batch, n] numpy array encoding initial value function.
       v: [batch, n] numpy array encoding initial value function.
    """

    T, batch_size, n, m = B.shape

    K = jnp.zeros((T, batch_size, m, n))
    k = jnp.zeros((T, m))

    Qf = sp.linalg.block_diag(*Qf).reshape((batch_size, n, batch_size, n))

    def body(tt, inputs):
        K, k, V, v = inputs
        t = T - 1 - tt
        V, v, K_t, k_t = batch_lqr_step(
            V, v, Q[t], q[t], R[t], r[t], M[t], A[t], B[t], d[t]
        )
        K = jax.ops.index_update(K, jax.ops.index[t], K_t)
        k = jax.ops.index_update(k, jax.ops.index[t], k_t)

        return K, k, V, v

    return jax.lax.fori_loop(0, T, body, (K, k, Qf, qf))
