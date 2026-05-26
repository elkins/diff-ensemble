import jax
import jax.numpy as jnp
from ensemble_predictor.model import EnsembleVAE
from ensemble_predictor.train import create_train_state, train_step

def test_initialization():
    seq_len = 10
    latent_dim = 16
    ensemble_size = 5
    model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)
    
    rng = jax.random.PRNGKey(0)
    input_shape = (1, seq_len, 4) # 4 features per residue
    
    state = create_train_state(model, rng, learning_rate=1e-3, input_shape=input_shape)
    assert state.params is not None
    print("Initialization successful.")

def test_gradient_flow():
    seq_len = 10
    latent_dim = 16
    ensemble_size = 5
    model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)
    
    rng = jax.random.PRNGKey(0)
    batch_x = jnp.ones((1, seq_len, 4))
    
    # Dummy experimental data
    num_atoms = 3 * seq_len
    q_values = jnp.linspace(0.01, 0.5, 20)
    exp_saxs = jnp.ones(20)
    form_factors = jnp.ones((num_atoms, 20)) # Match new backbone atom count (N, Ca, C)
    
    state = create_train_state(model, rng, learning_rate=1e-3, input_shape=batch_x.shape)
    
    state, loss, bio_loss, kl_loss = train_step(state, batch_x, exp_saxs, q_values, form_factors)
    
    assert loss > 0
    print(f"Loss: {loss}, Bio Loss: {bio_loss}, KL Loss: {kl_loss}")
    print("Gradient flow test passed.")

if __name__ == "__main__":
    test_initialization()
    test_gradient_flow()
