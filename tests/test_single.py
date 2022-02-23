import blqr
import jax
import jax.numpy as jnp
import jax.random as jr
import chex
from fixtures import system_key, system_dimensions, system_matrices, batch_size

jax.config.update("jax_enable_x64", True)


def test_uncontrolled_rollout(system_dimensions, system_matrices):
    n, m, T = system_dimensions
    _, _, _, _, _, _, _, A, B, d = system_matrices

    key = jr.PRNGKey(40)
    key, subkey = jr.split(key)
    x0 = jr.normal(subkey, (n,))
    X = blqr.uncontrolled_rollout(x0, A, B, d)
    chex.assert_shape(X, (T + 1, n))


def test_lqr(system_dimensions, system_matrices):
    n, m, T = system_dimensions
    # generate random system matrices
    Qf, qf, Q, q, R, r, M, A, B, d = system_matrices
    key = jr.PRNGKey(40)
    chex.assert_shape(Qf, (n, n))
    chex.assert_shape(qf, (n,))
    chex.assert_shape(R, (T, m, m))
    chex.assert_shape(r, (T, m))
    chex.assert_shape(M, (T, n, m))
    chex.assert_shape(A, (T, n, n))
    chex.assert_shape(B, (T, n, m))
    chex.assert_shape(d, (T, n))

    # solve LQR problem
    K, k, V, v = blqr.lqr(Qf, qf, Q, q, R, r, M, A, B, d)
    chex.assert_shape(K, (T, m, n))
    chex.assert_shape(k, (T, m))
    chex.assert_shape(V, (n, n))
    chex.assert_shape(v, (n,))

    # simulate dynamics forward
    key, subkey = jr.split(key)
    x0 = jr.normal(subkey, (n,))
    X, U = blqr.rollout(K, k, x0, A, B, d)
    chex.assert_shape(X, (T + 1, n))
    chex.assert_shape(U, (T, m))
