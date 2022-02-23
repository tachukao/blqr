import blqr
import jax
import jax.numpy as jnp
import jax.random as jr
import chex
from fixtures import (
    system_key,
    system_dimensions,
    system_matrices,
    batch_size,
    batch_system_matrices,
    blocked_batch_system_matrices,
)


def test_uncontrolled_rollout(system_dimensions, batch_size, batch_system_matrices):
    n, m, T = system_dimensions
    # generate random system matrices
    _, _, _, _, _, _, _, A, B, d = batch_system_matrices

    key = jr.PRNGKey(40)
    key, subkey = jr.split(key)
    x0 = jr.normal(subkey, (n,))
    X = blqr.batch_uncontrolled_rollout(x0, A, B, d)
    chex.assert_shape(X, (T + 1, batch_size, n))


def test_blqr_step(
    system_dimensions, batch_size, batch_system_matrices, blocked_batch_system_matrices
):
    """Check that results are the same passing blqr_step and lqr_step."""
    n, m, T = system_dimensions
    Qf, qf, Q, q, R, r, M, A, B, d = batch_system_matrices
    bQf, bqf, bQ, bq, bR, br, bM, bA, bB, bd = blocked_batch_system_matrices

    # check blocked batched system matrices
    chex.assert_shape(bQf, (batch_size * n, batch_size * n))
    chex.assert_shape(bqf, (batch_size * n,))
    chex.assert_shape(bR, (T, m, m))
    chex.assert_shape(br, (T, m))
    chex.assert_shape(bM, (T, batch_size * n, m))
    chex.assert_shape(bA, (T, batch_size * n, batch_size * n))
    chex.assert_shape(bB, (T, batch_size * n, m))
    chex.assert_shape(bd, (T, batch_size * n))

    bV, bv, bK, bk = blqr.lqr_step(
        bQf, bqf, bQ[-1], bq[-1], bR[-1], br[-1], bM[-1], bA[-1], bB[-1], bd[-1]
    )

    chex.assert_shape(bV, (batch_size * n, batch_size * n))
    chex.assert_shape(bv, (batch_size * n,))
    chex.assert_shape(bK, (m, batch_size * n))
    chex.assert_shape(bk, (m,))

    # TODO: now we try and do blqr_step
    V, v, K, k = blqr.batch_lqr_step(
        # reshape Qf to have the same dimensions as V
        bQf.reshape((batch_size, n, batch_size, n)),
        qf,
        Q[-1],
        q[-1],
        R[-1],
        r[-1],
        M[-1],
        A[-1],
        B[-1],
        d[-1],
    )

    assert jnp.allclose(K.ravel(), bK.ravel())
    assert jnp.allclose(k.ravel(), bk.ravel())
    assert jnp.allclose(V.ravel(), bV.ravel())
    assert jnp.allclose(v.ravel(), bv.ravel())


def test_batch_lqr(
    system_dimensions, batch_size, batch_system_matrices, blocked_batch_system_matrices
):
    n, m, T = system_dimensions
    Qf, qf, Q, q, R, r, M, A, B, d = batch_system_matrices

    key = jr.PRNGKey(40)
    chex.assert_shape(Qf, (batch_size, n, n))
    chex.assert_shape(qf, (batch_size, n))
    chex.assert_shape(R, (T, m, m))
    chex.assert_shape(r, (T, m))
    chex.assert_shape(M, (T, batch_size, n, m))
    chex.assert_shape(A, (T, batch_size, n, n))
    chex.assert_shape(B, (T, batch_size, n, m))
    chex.assert_shape(d, (T, batch_size, n))

    # solve LQR problem
    K, k, V, v = blqr.batch_lqr(Qf, qf, Q, q, R, r, M, A, B, d)
    chex.assert_shape(K, (T, batch_size, m, n))
    chex.assert_shape(k, (T, m))
    chex.assert_shape(V, (batch_size, n, batch_size, n))
    chex.assert_shape(v, (batch_size, n))

    # check that blqr give the same results as lqr
    bQf, bqf, bQ, bq, bR, br, bM, bA, bB, bd = blocked_batch_system_matrices
    bK, bk, bV, bv = blqr.lqr(bQf, bqf, bQ, bq, bR, br, bM, bA, bB, bd)

    assert jnp.allclose(V.ravel(), bV.ravel())
    assert jnp.allclose(v.ravel(), bv.ravel())
    assert jnp.allclose(K.ravel(), bK.ravel())
    assert jnp.allclose(k.ravel(), bk.ravel())

    key, subkey = jr.split(key)
    x0 = jr.normal(subkey, (batch_size, n))
    # let's rollout the trajectories and check that they are the same as that of lqr

    X, U = blqr.batch_rollout(K, k, x0, A, B, d)
    bx0 = x0.ravel()
    bX, bU = blqr.rollout(bK, bk, bx0, bA, bB, bd)
    assert jnp.allclose(X.ravel(), bX.ravel())
    assert jnp.allclose(U.ravel(), bU.ravel())
