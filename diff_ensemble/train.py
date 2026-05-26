import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from diff_ensemble.model import EnsembleVAE
from diff_ensemble.observables import get_ensemble_saxs, kld_loss, biophysical_loss

class TrainState(train_state.TrainState):
    key: jax.random.PRNGKey

def create_train_state(model, rng, learning_rate, input_shape):
    """Initializes the training state."""
    params = model.init(rng, jnp.ones(input_shape), rng)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, key=rng)

@jax.jit
def train_step(state, batch_x, exp_saxs, q_values, form_factors, beta=0.1):
    """Performs a single training step."""
    def loss_fn(params):
        torsions, mean, logvar = state.apply_fn(
            {'params': params}, batch_x, state.key)
        
        # Convert torsions to coords (simplified for MVP)
        # We use a method on the module, but since we are using apply_fn,
        # we might need to be careful. For now, let's assume EnsembleVAE
        # has a separate logic or we call the method manually.
        # Actually, let's instantiate the model to call generate_coordinates.
        model = EnsembleVAE(
            seq_len=batch_x.shape[1], 
            latent_dim=mean.shape[-1], 
            ensemble_size=torsions.shape[0]
        )
        coords = model.generate_coordinates(torsions)
        
        pred_saxs = get_ensemble_saxs(coords, q_values, form_factors)
        
        bio_loss = biophysical_loss(pred_saxs, exp_saxs)
        kl_loss = kld_loss(mean, logvar)
        
        total_loss = bio_loss + beta * kl_loss
        return total_loss, (bio_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (bio_loss, kl_loss)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    # Update RNG key
    new_key, _ = jax.random.split(state.key)
    state = state.replace(key=new_key)
    
    return state, loss, bio_loss, kl_loss
