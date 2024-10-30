import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
jax.config.update("jax_enable_x64", True)
from quadax import quadgk
from scipy.integrate import odeint
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from astropy import units as u 
from astropy import constants as const
import time


__all__ = ["Xklm", 
           "Xklm_vec",
           "TidalModelAveraged", 
           "TidalModelAveragedPlanar"]


@jax.jit
def xint(E, ecc, k, l, m):
    f = 2 * jnp.arctan(jnp.sqrt((1+ecc)/(1-ecc)) * jnp.tan(E/2))
    xint = 1/jnp.pi * (1 - ecc*jnp.cos(E))**(l+1) * jnp.cos(m*f - k*(E - ecc*jnp.sin(E)))
    return xint

@jax.jit
def Xklm(ecc, k=0, l=-3, m=2):
    integrand = partial(xint, ecc=ecc, k=k, l=l, m=m)
    return quadgk(integrand, [0., jnp.pi])[0]

# Vectorize the function using jax.vmap
Xklm_vec = jax.vmap(Xklm, in_axes=(None, 0, None, None))


class TidalModelBase(object):

    def __init__(self, 
                 params: dict, 
                 states: dict, 
                 nterms=5):

        self.default_params = {
                "mass1": 1.0 * u.Msun,
                "mass2": 1.0 * u.Msun,
                "radius1": 1.0 * u.Rsun,
                "radius2": 1.0 * u.Rsun,
                "rad_gyr1": 2/5 * u.dimensionless_unscaled,
                "rad_gyr2": 2/5 * u.dimensionless_unscaled,
                "tidal_tau": 1 * u.s,
                "tidal_Q": 1e6 * u.dimensionless_unscaled,
                "Gconst": const.G.si.value * u.m**3 / u.kg / u.s**2,
                "kf": 3/2 * u.dimensionless_unscaled,
                "mean_motion_init": 2*jnp.pi / (10 * u.day),
        }

        self.default_states = {
                "ecc": 0.2 * u.dimensionless_unscaled,
                "omega1": 2*jnp.pi / (1 * u.day),
                "omega2": 2*jnp.pi / (.5 * u.day),
                "obliquity1": 0.0 * u.deg,
                "obliquity2": 0.0 * u.deg,
        }

        # states = self.check_states(states)

        self.param_keys = list(params.keys())
        self.state_keys = list(states.keys())

        # number of terms used in series expansion
        self.nterms = nterms

        # turn input parameters into class attributes and convert to SI units
        for key, val in params.items():
            setattr(self, key, val.si.value)

        states_init = []
        for key, val in states.items():
            setattr(self, key, val.si.value)
            states_init.append(val.si.value)

        # save initial conditions
        self.states_dict_init = states
        self.states_init = jnp.array(states_init)

        # Compute intermediate quantities
        self.mass_tot = self.mass1 + self.mass2
        self.beta = self.mass1 * self.mass2 / self.mass_tot  # beta in CV2023
        self.mu = self.Gconst * self.mass_tot  # mu in CV2023
        self.alpha = self.mass1 * self.mass2 * self.Gconst**(2/3) / (self.mass1 + self.mass2)**(1/3)
        self.Imom1 = self.rad_gyr1 * self.mass1 * self.radius1**2 
        self.Imom2 = self.rad_gyr2 * self.mass2 * self.radius2**2 

        # compute other orbital elements, mean_motion_init sets the total angular momentum of the system
        self.semi_major_axis_init = (self.mu / self.mean_motion_init**2)**(1/3)

        # Total angular momentum of the system initial conditions 
        self.Jrot1_init = self.Imom1 * self.omega1 
        self.Jrot2_init = self.Imom2 * self.omega2
        self.Jorb_init = self.alpha * self.mean_motion_init**(-1/3) * (1 - self.ecc**2)**(1/2)
        self.Jtot_init = self.Jrot1_init + self.Jrot2_init + self.Jorb_init


    def check_states(self, states_dict):

        required_states = ["ecc", "omega1", "omega2", "obliquity1", "obliquity2"]

        # fill in missing initial conditions with default values
        for key in required_states:
            if key not in states_dict.keys():
                states_dict[key] = self.default_states[key]
                print(f"Initial condition {key} not specified. Using default value {default_states[key]}")
    
        return states_dict


    @partial(jit, static_argnums=(0,))
    def orbital_elements(self, ecc: jnp.float64, omega1: jnp.float64, omega2: jnp.float64):

        # Compute mean motion and semi-major axis
        # note: assuming that the total angular momentum of the system is conserved
        mean_motion = (self.alpha * jnp.sqrt(1 - ecc**2) / (self.Jtot_init - self.Imom1 * omega1 - self.Imom2 * omega2))**3
        semi_major_axis = (self.mu / mean_motion**2)**(1/3)

        return mean_motion, semi_major_axis


    @partial(jit, static_argnums=(0,))
    def Imk2(self, sigma: jnp.float64):

        raise NotImplementedError

    
    @partial(jit, static_argnums=(0,))
    def Rek2(self, sigma: jnp.float64):

        raise NotImplementedError
    
    
    @partial(jit, static_argnums=(0,))
    def Imk2_vec(self, sigma: jnp.array):
            
        return jax.vmap(self.Imk2)(sigma)
    

    @partial(jit, static_argnums=(0,))
    def Rek2_vec(self, sigma: jnp.array):

        return jax.vmap(self.Rek2)(sigma)
    

    @partial(jit, static_argnums=(0,))
    def tidal_torque(self, times: jnp.array, state: jnp.array, args: dict):
            
        raise NotImplementedError
    

    @partial(jit, static_argnums=(0,))
    def integrate_solution(self, ts: jnp.array):
        """
        Integrate the tidal evolution of the system, return only final state
        """

        # initial conditions
        y0 = self.states_init
        self.y0 = y0

        # Integrate the ODEs using diffrax
        term = ODETerm(self.tidal_torque)
        solver = Tsit5()
        t0, t1, dt0 = ts[0], ts[-1], None
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

        saveat = SaveAt(t1=True)
        tstart = time.time()
        sol = diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat, args=(None,), stepsize_controller=stepsize_controller).ys
        print("integration time (s):", time.time() - tstart, "\n")

        return sol
    
    
    def integrate_evolution(self, ts, return_si=False):
        """
        Integrate the tidal evolution of the system, return states at each timestep
        """

        # integration output times
        self.ts = ts

        # initial conditions
        y0 = self.states_init
        self.y0 = y0

        # Integrate the ODEs using diffrax
        term = ODETerm(self.tidal_torque)
        solver = Tsit5()
        t0, t1, dt0 = ts[0], ts[-1], None
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

        saveat = SaveAt(ts=ts)
        tstart = time.time()
        sol = diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat, args=(None,), stepsize_controller=stepsize_controller).ys
        print("integration time (s):", time.time() - tstart, "\n")

        # get original units and unit types
        orig_units = [s.unit for s in self.states_dict_init.values()]
        orig_types = [u.get_physical_type(s) for s in orig_units]

        # dictionary to convert between physical types and units
        si_units = [u.s, u.Hz, u.kg, u.m, u.K, u.rad, u.dimensionless_unscaled]
        si_types = [u.get_physical_type(x) for x in si_units]
        si_dict = dict(zip(si_types, si_units))

        output = {self.state_keys[ii]: jnp.array(sol.T[ii]) * si_dict[orig_types[ii]] for ii in range(sol.shape[1])}

        # compute other orbital elements 
        output["mean_motion"], output["semi_major_axis"] = self.orbital_elements(output["ecc"].value, output["omega1"].value, output["omega2"].value)
        output["mean_motion"] *= u.Hz
        output["semi_major_axis"] *= u.m
        output["orbital_period"] = (2 * jnp.pi / output["mean_motion"])
        output["rotation_period1"] = (2 * jnp.pi / output["omega1"])
        output["rotation_period2"] = (2 * jnp.pi / output["omega2"])
        output["time"] = ts * u.s
        
        if return_si == True:
            return output
        else:
            output["time"] = output["time"].to(u.yr)
            output["omega1"] = output["omega1"].to(1 / u.day)
            output["omega2"] = output["omega2"].to(1 / u.day)
            output["mean_motion"] = output["mean_motion"].to(1 / u.day)
            output["semi_major_axis"] = output["semi_major_axis"].to(u.AU)
            output["orbital_period"] = output["orbital_period"].to(u.day)
            output["rotation_period1"] = output["rotation_period1"].to(u.day)
            output["rotation_period2"] = output["rotation_period2"].to(u.day)

            return output


class TidalModelAveraged(TidalModelBase):
    
    def __init__(self, params, states, nterms=5):
        super().__init__(params, states, nterms=nterms)


    @partial(jit, static_argnums=(0,))
    def T1_avg(self, 
               body_state: jnp.array, 
               state: jnp.array, 
               T0: jnp.float64, 
               x:jnp.float64, 
               Xk_n3_n2: jnp.array, 
               Xk_n3_0: jnp.array, 
               Xk_n3_2: jnp.array, 
               k_values: jnp.array):

        _, omega, _ = body_state
        mean_motion, _ = self.orbital_elements(*state[:3])

        term1 = 9/32 * self.Imk2_vec(-k_values*mean_motion)
        term1 *= ((1 - x**2) * (Xk_n3_n2**2 - Xk_n3_2**2))

        term2 = 3/16 * self.Imk2_vec(omega - k_values*mean_motion)
        term2 *= (4*x**2 * Xk_n3_0**2 + (1 - x**2) * Xk_n3_n2**2 - (1 + x)**2 * (2 - x) * Xk_n3_2**2)

        term3 = 3/32 * self.Imk2_vec(2*omega - k_values*mean_motion)
        term3 *= (4*x * (1 - x**2) * Xk_n3_0**2 + (1 - x)**3 * (Xk_n3_n2**2 - (1 + x)**3 * Xk_n3_2**2))

        T1 = -T0 * jnp.sum(term1 + term2 + term3)
        
        return T1 
    

    @partial(jit, static_argnums=(0,))
    def T2_avg(self,
               body_state: jnp.array, 
               state: jnp.array, 
               T0: jnp.float64, 
               x:jnp.float64, 
               Xk_n3_n2: jnp.array, 
               Xk_n3_0: jnp.array, 
               Xk_n3_2: jnp.array, 
               k_values: jnp.array):

        _, omega, _ = body_state
        mean_motion, _ = self.orbital_elements(*state[:3])

        term1 = 9/32 * self.Imk2_vec(-k_values*mean_motion)
        term1 *= (x * (1 - x**2) * (Xk_n3_n2**2 - Xk_n3_2**2))

        term2 = 3/16 * self.Imk2_vec(omega - k_values*mean_motion)
        term2 *= (4*x**2 * Xk_n3_0**2 + (1 - x**2) * (1 + 2*x) * Xk_n3_n2**2 + (1 + x)**2 * (1 - 2*x) * Xk_n3_2**2)

        term3 = 3/32 * self.Imk2_vec(2*omega - k_values*mean_motion)
        term3 *= (4*(1 - x**2) * Xk_n3_0**2 + (1 - x)**3 * Xk_n3_n2**2 - (1 + x)**3 * Xk_n3_2**2)

        T2 = T0 * jnp.sum(term1 + term2 + term3)
       
        return T2
    

    @partial(jit, static_argnums=(0,))
    def T3_avg(self, 
               body_state: jnp.array, 
               state: jnp.array, 
               T0: jnp.float64, 
               x:jnp.float64, 
               Xk_n3_n2: jnp.array, 
               Xk_n3_0: jnp.array, 
               Xk_n3_2: jnp.array, 
               k_values: jnp.array):

        _, omega, _ = body_state
        mean_motion, _ = self.orbital_elements(*state[:3])

        term1 = 3/32 * x * self.Rek2_vec(-k_values*mean_motion)
        term1 *= (4 * (1 - 3*x**2) * Xk_n3_0**2 + 3*(1 - x**2) * (Xk_n3_n2**2 + Xk_n3_2**2))

        term2 = -3/16 * self.Rek2_vec(omega - k_values*mean_motion)
        term2 *= (4*x * (1 - 2*x**2) * Xk_n3_0**2 - (1 - x)**2 * (1 + 2*x) * Xk_n3_n2**2 + (1 + x)**2 * (1 - 2*x) * Xk_n3_2**2)

        term3 = 3/32 * self.Rek2_vec(2*omega - k_values*mean_motion)
        term3 *= (4*x * (1 - x**2) * Xk_n3_0**2 + (1 - x)**3 * Xk_n3_n2**2 - (1 + x) * Xk_n3_2**2)

        T3 = -T0 * jnp.sum(term1 + term2 + term3)

        return T3
    

    @partial(jit, static_argnums=(0,))
    def dEorb_avg(self, 
                  body_state: jnp.array, 
                  state: jnp.array, 
                  T0: jnp.float64, 
                  x:jnp.float64, 
                  Xk_n3_n2: jnp.array, 
                  Xk_n3_0: jnp.array, 
                  Xk_n3_2: jnp.array, 
                  k_values: jnp.array):

        _, omega, _ = body_state
        mean_motion, _ = self.orbital_elements(*state[:3])

        term1 = 1/64 * k_values * self.Imk2_vec(-k_values * mean_motion)
        term1 *= (4*(1 - 3*x**2)**2 * Xk_n3_0**2 + 9*(1 - x**2)**2 * (Xk_n3_n2**2 + Xk_n3_2**2))

        term2 = 3/16 * k_values * self.Imk2_vec(omega - k_values * mean_motion) * (1 - x**2)
        term2 *= (4*x**2 * Xk_n3_0**2 + (1 - x)**2 * Xk_n3_n2**2 + (1 + x)**2 * Xk_n3_2**2)

        term3 = 3/64 * k_values * self.Imk2_vec(2*omega - k_values * mean_motion)
        term3 *= (4*(1 - x**2)**2 * Xk_n3_0**2 + (1 - x)**4 * Xk_n3_n2**2 + (1 + x)**4 * Xk_n3_2**2)

        dEorb = mean_motion * T0 * jnp.sum(term1 + term2 + term3)

        return dEorb
    
    
    @partial(jit, static_argnums=(0,))
    def dEcc(self, 
             body_state: jnp.array, 
             state: jnp.array, 
             T0: jnp.float64, 
             T1: jnp.float64, 
             T2: jnp.float64, 
             x: jnp.float64, 
             Xk_n3_n2: jnp.array, 
             Xk_n3_0: jnp.array, 
             Xk_n3_2: jnp.array, 
             k_values: jnp.array):

        ecc, omega, obliquity = body_state

        # functions to parse if/else statement in jax.lax.cond
        def ecc_zero(_):
            return 0.0

        def ecc_not_zero(_):
            mean_motion, semi_major_axis = self.orbital_elements(*state[:3])
            dEorb = self.dEorb_avg(body_state, state, T0, x, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)
            return jnp.sqrt(1 - ecc*2) / (self.beta * self.mu * semi_major_axis**2 * ecc) \
                    * (jnp.sqrt(1 - ecc*2) / mean_motion * dEorb - T1 - T2 * x)

        decc = jax.lax.cond(ecc < 1e-10, ecc_zero, ecc_not_zero, operand=None)
        return decc
    

    @partial(jit, static_argnums=(0,))
    def tidal_torque(self, 
                     times: jnp.array, 
                     state: jnp.array,
                     args: dict):

        ecc, omega1, omega2, obliquity1, obliquity2 = state
        mean_motion, semi_major_axis = self.orbital_elements(*state[:3])

        # Compute hansen coefficients
        """
        Xk(-3,-2) = Xk_n3_n2
        Xk(-3,0) = Xk_n3_0
        Xk(-3,2) = Xk_n3_2
       """
        k_values = jnp.arange(-self.nterms, self.nterms)
        Xk_n3_n2 = Xklm_vec(ecc, k_values, -3, -2)
        Xk_n3_0 = Xklm_vec(ecc, k_values, -3, 0)
        Xk_n3_2 = Xklm_vec(ecc, k_values, -3, 2)

        # states of each individual body
        pri_state = jnp.array([ecc, omega1, obliquity1])
        sec_state = jnp.array([ecc, omega2, obliquity2])

        # intermediate quantities
        x_pri = jnp.cos(obliquity1)
        x_sec = jnp.cos(obliquity2)
        T0_pri = self.Gconst * self.mass2**2 * self.radius1**5 / semi_major_axis**6
        T0_sec = self.Gconst * self.mass1**2 * self.radius2**5 / semi_major_axis**6

        # Compute tidal torque terms
        T1_pri = self.T1_avg(pri_state, state, T0_pri, x_pri, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)
        T2_pri = self.T2_avg(pri_state, state, T0_pri, x_pri, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)

        T1_sec = self.T1_avg(sec_state, state, T0_sec, x_sec, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)
        T2_sec = self.T2_avg(sec_state, state, T0_sec, x_sec, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)

        # ODE derivatives
        ddt_ecc1 = self.dEcc(pri_state, state, T0_pri, T1_pri, T2_pri, x_pri, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)
        ddt_ecc2 = self.dEcc(sec_state, state, T0_sec, T1_sec, T2_sec, x_sec, Xk_n3_n2, Xk_n3_0, Xk_n3_2, k_values)
        ddt_ecc = ddt_ecc1 + ddt_ecc2
        
        ddt_omega1 = -(T1_pri * x_pri + T2_pri) / self.Imom1
        ddt_omega2 = -(T1_sec * x_sec + T2_sec) / self.Imom2

        ddt_obliquity1 = (T1_pri / (self.Imom1 * omega1) - T2_pri / (self.beta * jnp.sqrt(self.mu * semi_major_axis * (1 - ecc**2)))) * jnp.sin(obliquity1)
        ddt_obliquity2 = (T1_sec / (self.Imom2 * omega2) - T2_sec / (self.beta * jnp.sqrt(self.mu * semi_major_axis * (1 - ecc**2)))) * jnp.sin(obliquity2)

        return jnp.array([ddt_ecc, ddt_omega1, ddt_omega2, ddt_obliquity1, ddt_obliquity2])
    


class TidalModelAveragedPlanar(TidalModelBase):
    
    def __init__(self, params, states, nterms=5):
        super().__init__(params, states, nterms=nterms)


    def check_states(self, states_dict):

        required_states = ["ecc", "omega1", "omega2"]

        # fill in missing initial conditions with default values
        for key in required_states:
            if key not in states_dict.keys():
                states_dict[key] = self.default_states[key]
                print(f"Initial condition {key} not specified. Using default value {default_states[key]}")
    
        return states_dict


    @partial(jit, static_argnums=(0,))
    def dEcc(self, 
             body_state: jnp.array, 
             state: jnp.array, 
             E0: jnp.float64, 
             Xk_n3_0: jnp.array, 
             Xk_n3_2: jnp.array, 
             k_values: jnp.array):

        ecc, omega = body_state
        mean_motion, semi_major_axis = self.orbital_elements(*state[:3])

        # functions to parse if/else statement in jax.lax.cond
        def ecc_zero(_):
            return 0.0

        def ecc_not_zero(_):
            term1 = self.Imk2_vec(-k_values * mean_motion) * Xk_n3_0**2 * k_values * jnp.sqrt(1 - ecc**2)
            term2 = -3 * self.Imk2_vec(2 * omega - k_values * mean_motion) * Xk_n3_2**2 * (2 - k_values * jnp.sqrt(1 - ecc**2))
            decc = jnp.sum(term1 + term2)
            decc *= E0 * jnp.sqrt(1 - ecc**2) / (4 * ecc)
            return decc

        decc = jax.lax.cond(ecc < 1e-10, ecc_zero, ecc_not_zero, operand=None)
        return decc
    

    @partial(jit, static_argnums=(0,))
    def dOmega(self, 
               body_state: jnp.array, 
               state: jnp.array, 
               T0: jnp.float64, 
               Imom: jnp.float64, 
               Xk_n3_2: jnp.array, 
               k_values: jnp.array):

        ecc, omega = body_state
        mean_motion, semi_major_axis = self.orbital_elements(*state[:3])

        Imk2_values = self.Imk2_vec(2 * omega - k_values * mean_motion)
        domega = -3/2 * T0 / Imom * jnp.dot(Imk2_values, Xk_n3_2**2)

        return domega
    

    @partial(jit, static_argnums=(0,))
    def tidal_torque(self, 
                     times: jnp.array, 
                     state: jnp.array,
                     args: dict):

        ecc, omega1, omega2 = state
        mean_motion, semi_major_axis = self.orbital_elements(*state[:3])

        # Compute hansen coefficients
        """
        Xk(-3,-2) = Xk_n3_n2
        Xk(-3,0) = Xk_n3_0
        Xk(-3,2) = Xk_n3_2
       """
        k_values = jnp.arange(-self.nterms, self.nterms)
        Xk_n3_0 = Xklm_vec(ecc, k_values, -3, 0)
        Xk_n3_2 = Xklm_vec(ecc, k_values, -3, 2)

        # states of each individual body
        pri_state = jnp.array([ecc, omega1])
        sec_state = jnp.array([ecc, omega2])

        T0_pri = self.Gconst * self.mass2**2 * self.radius1**5 / semi_major_axis**6
        T0_sec = self.Gconst * self.mass1**2 * self.radius2**5 / semi_major_axis**6

        E0_pri = mean_motion * (self.mass2 / self.mass1) * (self.radius1 / semi_major_axis)**5
        E0_sec = mean_motion * (self.mass1 / self.mass2) * (self.radius2 / semi_major_axis)**5

        # ODE derivatives
        ddt_ecc  = self.dEcc(pri_state, state, E0_pri, Xk_n3_0, Xk_n3_2, k_values)
        ddt_ecc += self.dEcc(sec_state, state, E0_sec, Xk_n3_0, Xk_n3_2, k_values)
        
        ddt_omega1 = self.dOmega(pri_state, state, T0_pri, self.Imom1, Xk_n3_2, k_values)
        ddt_omega2 = self.dOmega(sec_state, state, T0_sec, self.Imom2, Xk_n3_2, k_values)

        return jnp.array([ddt_ecc, ddt_omega1, ddt_omega2])