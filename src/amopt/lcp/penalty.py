import numpy as np


def apply_boundary_conditions(option_type, V, S, K, r, q, T, n, dt):
    tau = T - n * dt

    if option_type.lower() == 'call':
        V[0] = 0
        V[-1] = S[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
    else:
        V[0] = K * np.exp(-r * tau)
        V[-1] = 0


def reset_thomas_arrays(c_star, d_star):
    c_star.fill(0.0)
    d_star.fill(0.0)


# Crank–Nicolson PDE solver with Thomas algorithm for tri-diagonal systems
def penalty_pde_solver(option_type,
                       L_imp, D_imp, U_imp,
                       L_exp, D_exp, U_exp,
                       K, spatial_grid,
                       T, r, q, time_intervals
                       ):
    dt = T / time_intervals

    # Penalty must be large otherwise lambda = 0 => European Option
    _lambda = 1e3
    tolerance = 1e-6

    R = np.zeros(len(spatial_grid))
    V_new = np.zeros(len(spatial_grid))
    c_star = np.zeros(len(spatial_grid))
    d_star = np.zeros(len(spatial_grid))
    V_S_t = np.zeros((len(spatial_grid), time_intervals))

    # STEP 1: Initialise option value at MATURITY i.e. when t = T, V = payoff - comparing element-wise
    if option_type.lower() == 'call':
        V = np.maximum(spatial_grid - K, 0.0)
        payoff = np.maximum(spatial_grid - K, 0.0)
    else:
        V = np.maximum(K - spatial_grid, 0.0)
        payoff = np.maximum(K - spatial_grid, 0.0)

    # Store maturity
    V_S_t[:, -1] = payoff.copy()

    # March backwards: solve for n = N-2 ... 0 (t=T-dt ... 0)
    for n in range(time_intervals - 2, -1, -1):

        # OLD solution is V^{n+1}
        V_old = V.copy()
        apply_boundary_conditions(option_type, V_old, spatial_grid, K, r, q, T, n + 1, dt)

        # NEW-time boundary values (time n) for implicit RHS injection
        V_new.fill(0.0)
        apply_boundary_conditions(option_type, V_new, spatial_grid, K, r, q, T, n, dt)
        bc_lo = V_new[0]
        bc_hi = V_new[-1]

        # Initial guess for V^n: start from old solution (good warm start)
        V_guess = V_old.copy()

        constraint = np.inf
        while constraint > tolerance:

            V_guess_old = V_guess.copy()

            # Exercise indicator based on current guess at time n
            exercise = (payoff > V_guess).astype(float)
            exercise[0] = 0.0
            exercise[-1] = 0.0

            # Build RHS using OLD solution (time n+1)
            R.fill(0.0)
            for k in range(1, len(spatial_grid) - 1):
                R[k] = (
                        L_exp[k] * V_old[k - 1]
                        + D_exp[k] * V_old[k]
                        + U_exp[k] * V_old[k + 1]
                        + _lambda * dt * payoff[k] * exercise[k]
                )

            # Inject NEW-time BC into RHS (implicit operator boundary contribution)
            R[1] -= L_imp[1] * bc_lo
            R[-2] -= U_imp[-2] * bc_hi

            reset_thomas_arrays(c_star, d_star)

            # Thomas with penalised diagonal
            denom = D_imp[1] + _lambda * dt * exercise[1]
            d_star[1] = R[1] / denom
            c_star[1] = U_imp[1] / denom

            for i in range(2, len(spatial_grid) - 1):
                denom = (D_imp[i] + _lambda * dt * exercise[i]) - c_star[i - 1] * L_imp[i]
                c_star[i] = U_imp[i] / denom
                d_star[i] = (R[i] - d_star[i - 1] * L_imp[i]) / denom

            # Back substitution
            V_new.fill(0.0)
            V_new[-2] = d_star[-2]
            for i in range(len(spatial_grid) - 3, 0, -1):
                V_new[i] = d_star[i] - c_star[i] * V_new[i + 1]

            # Apply hard constraint and reapply BC
            V_new = np.maximum(V_new, payoff)
            V_new[0] = bc_lo
            V_new[-1] = bc_hi

            V_guess = V_new.copy()
            constraint = np.max(np.abs(V_guess - V_guess_old))

        # Store solution at time n
        V_S_t[:, n] = V_guess
        V = V_guess.copy()

    # time is marching backwards hence we reverse the time column so that we are going forwards in time

    # Returning a 2D array of option values against stock prices and time
    return V_S_t
