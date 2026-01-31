import numpy as np


def apply_boundary_conditions(option_type, V, S, K, r, q, T, n, dt):
    tau = (T - n * dt)

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
def time_marching_pde_solver(option_type, L_imp, D_imp, U_imp, L_exp, D_exp, U_exp,
                             K, spatial_grid, T, r, q, time_intervals):
    dt = T / time_intervals

    V_S_t = np.zeros((len(spatial_grid), time_intervals))
    R = np.zeros(len(spatial_grid))
    V_new = np.zeros(len(spatial_grid))
    c_star = np.zeros(len(spatial_grid))
    d_star = np.zeros(len(spatial_grid))

    # STEP 1: Initialise option value at maturity (payoff) - comparing element-wise
    if option_type.lower() == 'call':
        V = np.maximum(spatial_grid - K, 0.0)
    else:
        V = np.maximum(K - spatial_grid, 0.0)

    # Store maturity explicitly
    V_S_t[:, -1] = V.copy()

    # STEP 2: March backwards in time using Crank–Nicolson
    # We solve for n = time_intervals-2 ... 0
    for n in range(time_intervals - 2, -1, -1):

        # V is the OLD solution at time n+1, so apply BC at n+1
        apply_boundary_conditions(option_type, V, spatial_grid, K, r, q, T, n + 1, dt)

        # New-time boundary values (time n) will be needed for the implicit RHS adjustment
        V_new.fill(0.0)
        apply_boundary_conditions(option_type, V_new, spatial_grid, K, r, q, T, n, dt)
        bc_lo = V_new[0]
        bc_hi = V_new[-1]

        # STEP 4: Reset RHS vector for current time step
        R.fill(0.0)

        for k in range(1, len(spatial_grid) - 1):
            # STEP 5: Build RHS using explicit Crank–Nicolson operator (interior nodes)
            R[k] = (L_exp[k] * V[k - 1]) + (D_exp[k] * V[k]) + (U_exp[k] * V[k + 1])

        R[1] -= L_imp[1] * bc_lo
        R[-2] -= U_imp[-2] * bc_hi

        # STEP 6: Reset Thomas algorithm work arrays
        reset_thomas_arrays(c_star, d_star)

        # STEP 7: Initialise Thomas algorithm at first interior node
        d_star[1] = R[1] / D_imp[1]
        c_star[1] = U_imp[1] / D_imp[1]

        for i in range(2, len(spatial_grid) - 1):
            # STEP 8: Thomas forward elimination (remove lower diagonal)
            denom = D_imp[i] - c_star[i - 1] * L_imp[i]
            c_star[i] = U_imp[i] / denom
            d_star[i] = (R[i] - d_star[i - 1] * L_imp[i]) / denom

        # STEP 9: Clear new solution vector before back substitution
        V_new.fill(0.0)

        # STEP 10: Thomas back substitution (solve upper-triangular system)
        for i in range(len(spatial_grid) - 2, 0, -1):
            V_new[i] = d_star[i] - c_star[i] * V_new[i + 1]

        apply_boundary_conditions(option_type, V_new, spatial_grid, K, r, q, T, n, dt)

        V_S_t[:, n] = V_new

        # STEP 12: Update solution for next time step
        V = V_new.copy()

    return V_S_t
