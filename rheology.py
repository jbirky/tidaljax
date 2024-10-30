from functools import partial
import jax.numpy as jnp
from jax import jit
from model import *

__all__ = ["TidalModelAveragedCTL", 
           "TidalModelAveragedPlanarCTL",
           "TidalModelAveragedCPL", 
           "TidalModelAveragedPlanarCPL"]


# =======================================================
# Constant time lag model
# ======================================================= 

class TidalModelAveragedCTL(TidalModelAveraged):
                
    def __init__(self, params, states, nterms=5):
        super().__init__(params, states, nterms=nterms)

    @partial(jit, static_argnums=(0,))
    def Rek2(self, sigma: jnp.float64):
        return self.kf

    @partial(jit, static_argnums=(0,))
    def Imk2(self, sigma: jnp.float64):
        return self.kf * sigma * self.tidal_tau
   

class TidalModelAveragedPlanarCTL(TidalModelAveragedPlanar):
                
    def __init__(self, 
                 params, 
                 states, 
                 nterms=5):
        
        super().__init__(params, states, nterms=nterms)

    @partial(jit, static_argnums=(0,))
    def Rek2(self, sigma: jnp.float64):
        return self.kf

    @partial(jit, static_argnums=(0,))
    def Imk2(self, sigma: jnp.float64):
        return self.kf * sigma * self.tidal_tau
    

# =======================================================
# Constant phase lag model
# ======================================================= 

class TidalModelAveragedCPL(TidalModelAveraged):
                
    def __init__(self, params, states, nterms=5):
        super().__init__(params, states, nterms=nterms)

    @partial(jit, static_argnums=(0,))
    def Rek2(self, sigma: jnp.float64):
        return self.kf

    @partial(jit, static_argnums=(0,))
    def Imk2(self, sigma: jnp.float64):
        return self.kf * jnp.sign(sigma) / self.tidal_Q
    

class TidalModelAveragedPlanarCPL(TidalModelAveragedPlanar):
                
    def __init__(self, params, states, nterms=5):
        super().__init__(params, states, nterms=nterms)

    @partial(jit, static_argnums=(0,))
    def Rek2(self, sigma: jnp.float64):
        return self.kf

    @partial(jit, static_argnums=(0,))
    def Imk2(self, sigma: jnp.float64):
        return self.kf * jnp.sign(sigma) / self.tidal_Q