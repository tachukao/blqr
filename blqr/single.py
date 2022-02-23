"""Finite-time linear quadratic regulator (LQR)

Adapted from https://github.com/google/trajax/blob/main/trajax/tvlqr.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp
import jax.scipy as sp


@jax.jit
def uncontrolled_rollout(x0, A, B, d):
    """Uncontrolled roll-out time-varying dynamics x[t+1] = A[t] x[t] + d[t]."""
    T, n, m = B.shape
    X = jnp.zeros((T + 1, n))
    X = jax.ops.index_update(X, jax.ops.index[0], x0)

    def body(t, X):
        x = jnp.matmul(A[t], X[t]) + d[t]
        X = jax.ops.index_update(X, jax.ops.index[t + 1], x)
        return X

    return jax.lax.fori_loop(0, T, body, X)


@jax.jit
def rollout(K, k, x0, A, B, d):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""

    T, m, n = K.shape
    X = jnp.zeros((T + 1, n))
    U = jnp.zeros((T, m))
    X = jax.ops.index_update(X, jax.ops.index[0], x0)

    def body(t, inputs):
        X, U = inputs
        u = jnp.matmul(K[t], X[t]) + k[t]
        x = jnp.matmul(A[t], X[t]) + jnp.matmul(B[t], u) + d[t]
        X = jax.ops.index_update(X, jax.ops.index[t + 1], x)
        U = jax.ops.index_update(U, jax.ops.index[t], u)
        return X, U

    return jax.lax.fori_loop(0, T, body, (X, U))


@jax.jit
def lqr_step(V, v, Q, q, R, r, M, A, B, d, delta=1e-8):
    """Single LQR Step.
    Args:
      V: [n, n] numpy array.
      v: [n] numpy array.
      Q: [n, n] numpy array.
      q: [n] numpy array.
      R: [m, m] numpy array.
      r: [m] numpy array.
      M: [n, m] numpy array.
      A: [n, n] numpy array.
      B: [n, m] numpy array.
      d: [n] numpy array.
      delta: Enforces positive definiteness by ensuring smallest eigenval > delta.
    Returns:
      V, v: updated matrices encoding quadratic value function.
      K, k: state feedback gain and affine term.
    """
    symmetrize = lambda x: (x + x.T) / 2

    AtV = jnp.matmul(A.T, V)
    AtVA = symmetrize(jnp.matmul(AtV, A))
    BtV = jnp.matmul(B.T, V)
    BtVA = jnp.matmul(BtV, A)

    G = symmetrize(R + jnp.matmul(BtV, B))
    # make G positive definite so that smallest eigenvalue > delta.
    S, _ = jnp.linalg.eigh(G)
    G_ = G + jnp.maximum(0.0, delta - S[0]) * jnp.eye(G.shape[0])

    H = BtVA + M.T
    h = jnp.matmul(B.T, v) + jnp.matmul(BtV, d) + r

    K = -sp.linalg.solve(G_, H, sym_pos=True)
    k = -sp.linalg.solve(G_, h, sym_pos=True)

    H_GK = H + jnp.matmul(G, K)
    V = symmetrize(Q + AtVA + jnp.matmul(H_GK.T, K) + jnp.matmul(K.T, H))
    v = (
        q
        + jnp.matmul(A.T, v)
        + jnp.matmul(AtV, d)
        + jnp.matmul(H_GK.T, k)
        + jnp.matmul(K.T, h)
    )

    return V, v, K, k


@jax.jit
def lqr(Qf, qf, Q, q, R, r, M, A, B, d):
    """Discrete-time Finite Horizon Time-varying LQR.
    Args:
      Qf: [n, n] numpy array.
      qf: [n] numpy array.
       Q: [T, n, n] numpy array.
       q: [T, n] numpy array.
       R: [T, m, m] numpy array.
       r: [T, m] numpy array.
       M: [T, n, m] numpy array.
       A: [T, n, n] numpy array.
       B: [T, n, m] numpy array.
       d: [T, n] numpy array.
    Returns:
       K: [T, m, n] Gains
       k: [T, m] Affine terms (u_t = jnp.matmul(K[t],  x_t) + k[t])
       V: [n, n] numpy array encoding initial value function.
       v: [n] numpy array encoding initial value function.
    """

    T = Q.shape[0]
    m = R.shape[1]
    n = Q.shape[1]

    K = jnp.zeros((T, m, n))
    k = jnp.zeros((T, m))

    def body(tt, inputs):
        K, k, V, v = inputs
        t = T - 1 - tt
        V, v, K_t, k_t = lqr_step(V, v, Q[t], q[t], R[t], r[t], M[t], A[t], B[t], d[t])
        K = jax.ops.index_update(K, jax.ops.index[t], K_t)
        k = jax.ops.index_update(k, jax.ops.index[t], k_t)

        return K, k, V, v

    return jax.lax.fori_loop(0, T, body, (K, k, Qf, qf))


