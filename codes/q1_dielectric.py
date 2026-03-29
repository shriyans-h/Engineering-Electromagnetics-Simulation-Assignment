"""
To run just the dielectric case without comparing, set RUN_UNIFORM=False.
"""

import numpy as np
import matplotlib.pyplot as plt

RUN_UNIFORM = True     # set False to skip re-solving the κ=1 case

# 1.  Grid & physical parameters  (same as Part A)
N   = 120
L   = 5e-3
h   = L / (N - 1)
x   = np.linspace(-L/2, L/2, N)
y   = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

V_A, V_B = 5.0, -5.0
y_A, y_B =  (5/3)*1e-3, -(5/3)*1e-3
j_A = int(np.argmin(np.abs(y - y_A)))
j_B = int(np.argmin(np.abs(y - y_B)))

# Interface row (y = 0)
j_int = int(np.argmin(np.abs(y - 0.0)))

# Dielectric constants
kappa1 = 4.0    # y > 0  (above interface)
kappa2 = 1.0    # y < 0  (below interface)

# 2.  Boundary-condition helper  (unchanged)
def apply_bc(V):
    V[0,  :]  = 0.0;  V[-1, :]  = 0.0
    V[:,  0]  = 0.0;  V[:, -1]  = 0.0
    V[j_A, :] = V_A;  V[j_B, :] = V_B

# 3.  Solver for uniform dielectric (κ = 1)
def solve_uniform(n_iter=5000):
    """Standard Jacobi solver for ∇²V = 0."""
    V = np.zeros((N, N))
    apply_bc(V)
    for _ in range(n_iter):
        V[1:-1, 1:-1] = 0.25 * (
            V[2:,  1:-1] + V[:-2, 1:-1] +
            V[1:-1, 2:]  + V[1:-1, :-2]
        )
        apply_bc(V)
    return V

# 4.  Solver for dielectric interface  ∇·(κ ∇V) = 0
def solve_dielectric(n_iter=5000):
    """
    Modified Jacobi for a horizontal dielectric interface at y = 0.
    """
    kax   = (kappa1 + kappa2) / 2.0   # in-plane averaging weight
    denom = 2.0 * kax + kappa1 + kappa2

    V = np.zeros((N, N))
    apply_bc(V)

    for _ in range(n_iter):
        # ── Interior of κ₁ region (j > j_int) ──────────────────────────────
        if j_int + 1 < N - 1:
            V[j_int+1:-1, 1:-1] = 0.25 * (
                V[j_int+2:,  1:-1] + V[j_int:-2, 1:-1] +
                V[j_int+1:-1, 2:]  + V[j_int+1:-1, :-2]
            )

        # ── Interior of κ₂ region (j < j_int) ──────────────────────────────
        if j_int - 1 > 0:
            V[1:j_int, 1:-1] = 0.25 * (
                V[2:j_int+1, 1:-1] + V[0:j_int-1, 1:-1] +
                V[1:j_int,    2:]  + V[1:j_int,   :-2]
            )

        # ── Interface row j_int: modified update ────────────────────────────
        V[j_int, 1:-1] = (
            kax  * (V[j_int, 2:] + V[j_int, :-2]) +
            kappa1 * V[j_int+1, 1:-1] +
            kappa2 * V[j_int-1, 1:-1]
        ) / denom

        apply_bc(V)
    return V

# 5.  Run solvers
print("Solving dielectric-interface case …")
V_diel = solve_dielectric(n_iter=5000)
print("  Done.")

if RUN_UNIFORM:
    print("Solving uniform-κ case for comparison …")
    V_uni = solve_uniform(n_iter=5000)
    print("  Done.")

def grad_field(V):
    dVdy, dVdx = np.gradient(V, h, h)
    return -dVdx, -dVdy

Ex_d, Ey_d = grad_field(V_diel)
E_mag_d     = np.sqrt(Ex_d**2 + Ey_d**2)

if RUN_UNIFORM:
    Ex_u, Ey_u = grad_field(V_uni)
    E_mag_u     = np.sqrt(Ex_u**2 + Ey_u**2)

# 6.  Plot A – Equipotential comparison
ncols = 2 if RUN_UNIFORM else 1
fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 7))

def plot_equipotential(ax, V_, title_):
    lv = np.linspace(V_B, V_A, 50)
    cf = ax.contourf(X*1e3, Y*1e3, V_, levels=lv, cmap='RdBu_r')
    ax.contour(X*1e3, Y*1e3, V_, levels=20, colors='k', linewidths=0.5)
    plt.colorbar(cf, ax=ax, label='V  (V)', fraction=0.046)
    ax.axhline(y_A*1e3, color='red',  lw=2, label='Plate A')
    ax.axhline(y_B*1e3, color='blue', lw=2, label='Plate B')
    # Mark the dielectric interface
    ax.axhline(0, color='lime', lw=1.5, ls='--', label='Interface y=0')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(title_, fontweight='bold')
    ax.legend(fontsize=8); ax.set_aspect('equal')

if RUN_UNIFORM:
    ax_list = axes
    plot_equipotential(ax_list[0], V_uni,  'Equipotential – κ = 1 (uniform)')
    plot_equipotential(ax_list[1], V_diel, f'Equipotential – κ₁={kappa1} (y>0), κ₂={kappa2} (y<0)')
else:
    plot_equipotential(axes, V_diel, f'Equipotential – κ₁={kappa1} (y>0), κ₂={kappa2} (y<0)')

plt.tight_layout()
plt.savefig('q1_diel_equipotential.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q1_diel_equipotential.png")

# 7.  Plot B – E-field lines comparison
fig2, axes2 = plt.subplots(1, ncols, figsize=(7*ncols, 7))

def plot_efield(ax, Ex_, Ey_, Em_, title_):
    lw = 2.0 * Em_ / (Em_.max() + 1e-12)
    strm = ax.streamplot(X*1e3, Y*1e3, Ex_, Ey_,
                         density=2.0, linewidth=lw,
                         color=Em_, cmap='inferno', arrowsize=1.0)
    plt.colorbar(strm.lines, ax=ax, label='|E| (V/m)', fraction=0.046)
    ax.axhline(y_A*1e3, color='red',  lw=2)
    ax.axhline(y_B*1e3, color='blue', lw=2)
    ax.axhline(0, color='lime', lw=1.5, ls='--', label='Interface y=0')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(title_, fontweight='bold')
    ax.legend(fontsize=8); ax.set_aspect('equal')

if RUN_UNIFORM:
    plot_efield(axes2[0], Ex_u, Ey_u, E_mag_u, 'E-field Lines – κ = 1')
    plot_efield(axes2[1], Ex_d, Ey_d, E_mag_d,
                f'E-field Lines – κ₁={kappa1} / κ₂={kappa2}')
else:
    plot_efield(axes2, Ex_d, Ey_d, E_mag_d,
                f'E-field Lines – κ₁={kappa1} / κ₂={kappa2}')

plt.tight_layout()
plt.savefig('q1_diel_efield_lines.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q1_diel_efield_lines.png")

# 8.  Plot C – |E| midline comparison
i_mid = N // 2
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(y*1e3, E_mag_d[:, i_mid], 'r-',  lw=2,
         label=f'κ₁={kappa1} (y>0), κ₂={kappa2} (y<0)')
if RUN_UNIFORM:
    ax3.plot(y*1e3, E_mag_u[:, i_mid], 'b--', lw=2, label='κ = 1 (uniform)')
ax3.axvline(y_A*1e3, color='k',   ls=':', lw=1.5, label='Plate A')
ax3.axvline(y_B*1e3, color='grey',ls=':', lw=1.5, label='Plate B')
ax3.axvline(0, color='lime', ls='--', lw=1.5, label='Interface y=0')
ax3.set_xlabel('y (mm)'); ax3.set_ylabel('|E| (V/m)')
ax3.set_title('|E| Along Vertical Midline – Dielectric Comparison',
              fontweight='bold')
ax3.legend(); ax3.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q1_diel_midline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q1_diel_midline.png")
