import blqr
import jax
import jax.numpy as jnp
import jax.scipy as sp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def system_dimensions():
    n = 2
    m = 1
    T = 30
    return n, m, T


@pytest.fixture
def batch_size():
    return 5


@pytest.fixture
def system_key():
    return jr.PRNGKey(10000)


@pytest.fixture
def system_matrices(system_key, system_dimensions):
    """Generate random system matrices."""
    n, m, T = system_dimensions
    key, subkey = jr.split(system_key)
    A = jr.normal(subkey, (T, n, n))
    key, subkey = jr.split(system_key)
    B = jr.normal(subkey, (T, n, m))
    key, subkey = jr.split(system_key)
    d = jr.normal(subkey, (T, n))
    Qf = jnp.eye(n)
    qf = -0.5 * jnp.ones(n)
    Q = jnp.tile(jnp.eye(n), (T, 1, 1))
    q = -0.5 * jnp.tile(jnp.ones(n), (T, 1))
    R = jnp.tile(jnp.eye(m), (T, 1, 1))
    r = -0.5 * jnp.tile(jnp.ones(m), (T, 1))
    M = -0.5 * jnp.tile(jnp.ones((n, m)), (T, 1, 1))
    return Qf, qf, Q, q, R, r, M, A, B, d


@pytest.fixture
def batch_system_matrices(system_key, batch_size, system_dimensions):
    """Generate batched random system matrices."""
    n, m, T = system_dimensions
    key, subkey = jr.split(system_key)
    A = jr.normal(subkey, (T, batch_size, n, n))
    key, subkey = jr.split(system_key)
    B = jr.normal(subkey, (T, batch_size, n, m))
    key, subkey = jr.split(system_key)
    d = jr.normal(subkey, (T, batch_size, n))
    Qf = jnp.tile(jnp.eye(n), (batch_size, 1, 1))
    qf = jnp.tile(jnp.zeros(n), (batch_size, 1))
    Q = jnp.tile(jnp.eye(n), (T, batch_size, 1, 1))
    q = jnp.tile(jnp.zeros(n), (T, batch_size, 1))
    R = jnp.tile(jnp.eye(m), (T, 1, 1))
    r = jnp.tile(jnp.zeros(m), (T, 1))
    M = jnp.tile(jnp.zeros((n, m)), (T, batch_size, 1, 1))
    return Qf, qf, Q, q, R, r, M, A, B, d


@pytest.fixture
def blocked_batch_system_matrices(batch_system_matrices):
    Qf, qf, Q, q, R, r, M, A, B, d = batch_system_matrices
    block_diag = lambda x: sp.linalg.block_diag(*x)
    vblock_diag = jax.vmap(block_diag)
    vravel = jax.vmap(jnp.ravel)
    Qf = block_diag(Qf)
    qf = jnp.ravel(qf)
    Q = vblock_diag(Q)
    q = vravel(q)
    R = vblock_diag(R)
    r = vravel(r)
    M = M.reshape((M.shape[0], -1, M.shape[-1]))
    A = vblock_diag(A)
    B = B.reshape((B.shape[0], -1, B.shape[-1]))
    d = vravel(d)
    return Qf, qf, Q, q, R, r, M, A, B, d
