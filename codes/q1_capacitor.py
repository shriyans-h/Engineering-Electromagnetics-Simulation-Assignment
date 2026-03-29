import numpy as np
import matplotlib.pyplot as plt

# 1.  Grid & physical parameters
N   = 120                          # grid points per side  (N × N)
L   = 5e-3                         # domain full width = 5 mm  [m]
h   = L / (N - 1)                  # uniform grid spacing  [m]

x   = np.linspace(-L/2, L/2, N)   # x-axis  (m)
y   = np.linspace(-L/2, L/2, N)   # y-axis  (m)
X, Y = np.meshgrid(x, y)           # shape (N, N); Y[j,i] = y[j], X[j,i] = x[i]

# Plate voltages and y-positions
V_A =  5.0                         # Plate A  [V]
V_B = -5.0                         # Plate B  [V]
y_A =  (5/3) * 1e-3                # y = +5/3 mm
y_B = -(5/3) * 1e-3                # y = -5/3 mm

# Find row indices closest to each plate
j_A = int(np.argmin(np.abs(y - y_A)))
j_B = int(np.argmin(np.abs(y - y_B)))

# 2.  Boundary-condition helper
def apply_bc(V):
    """Enforce Dirichlet BCs on every iteration."""
    V[0,  :]  = 0.0   # bottom wall grounded
    V[-1, :]  = 0.0   # top    wall grounded
    V[:,  0]  = 0.0   # left   wall grounded
    V[:, -1]  = 0.0   # right  wall grounded
    V[j_A, :] = V_A   # Plate A: all columns, row j_A
    V[j_B, :] = V_B   # Plate B: all columns, row j_B

# 3.  Iterative solver  (Jacobi – nearest-neighbour averaging)
#     ∇²V = 0  ⟹  V[j,i] = ¼ (V[j+1,i] + V[j-1,i] + V[j,i+1] + V[j,i-1])
V = np.zeros((N, N))
apply_bc(V)

n_iter = 5000            # ≥ 1000 as required; 5000 gives well-converged result
for _ in range(n_iter):
    # Vectorised Jacobi update for all interior nodes simultaneously
    V[1:-1, 1:-1] = 0.25 * (
        V[2:,   1:-1] +    # north neighbour  (j+1)
        V[:-2,  1:-1] +    # south neighbour  (j-1)
        V[1:-1,  2:] +     # east  neighbour  (i+1)
        V[1:-1, :-2]       # west  neighbour  (i-1)
    )
    apply_bc(V)            # re-impose BCs after each full sweep

# 4.  Electric field  E = −∇V
#     np.gradient(V, h, h) → [∂V/∂y, ∂V/∂x]  (axis-0 = y, axis-1 = x)

dVdy, dVdx = np.gradient(V, h, h)
Ex = -dVdx
Ey = -dVdy
E_mag = np.sqrt(Ex**2 + Ey**2)

# 5.  Plot 1 – Equipotential contour map
fig1, ax1 = plt.subplots(figsize=(7, 7))

levels_fill = np.linspace(V_B, V_A, 60)
cf = ax1.contourf(X * 1e3, Y * 1e3, V, levels=levels_fill, cmap='RdBu_r')
cs = ax1.contour( X * 1e3, Y * 1e3, V, levels=20, colors='k',
                  linewidths=0.5, linestyles='solid')
plt.colorbar(cf, ax=ax1, label='Electric Potential  V  (V)', fraction=0.046)

# Mark the plates
ax1.axhline(y_A * 1e3, color='red',   lw=2.5, label=f'Plate A  (+{V_A} V)')
ax1.axhline(y_B * 1e3, color='blue',  lw=2.5, label=f'Plate B  ({V_B} V)')

ax1.set_xlabel('x  (mm)', fontsize=12)
ax1.set_ylabel('y  (mm)', fontsize=12)
ax1.set_title('Equipotential Contour Map – Parallel-Plate Capacitor (κ = 1)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_aspect('equal')
plt.tight_layout()
plt.savefig('q1_equipotential.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q1_equipotential.png")

# 6.  Plot 2 – Electric field lines
fig2, ax2 = plt.subplots(figsize=(7, 7))

# Normalise line-width to field magnitude for visual clarity
lw_scaled = 2.5 * E_mag / (E_mag.max() + 1e-12)

strm = ax2.streamplot(X * 1e3, Y * 1e3, Ex, Ey,
                      density=2.2, linewidth=lw_scaled,
                      color=E_mag, cmap='inferno', arrowsize=1.0)
plt.colorbar(strm.lines, ax=ax2, label='|E|  (V/m)', fraction=0.046)

# Overlay equipotentials faintly
ax2.contour(X * 1e3, Y * 1e3, V, levels=15,
            colors='cyan', linewidths=0.6, alpha=0.5)

ax2.axhline(y_A * 1e3, color='red',  lw=2.5, label='Plate A (+5 V)')
ax2.axhline(y_B * 1e3, color='blue', lw=2.5, label='Plate B (−5 V)')

ax2.set_xlabel('x  (mm)', fontsize=12)
ax2.set_ylabel('y  (mm)', fontsize=12)
ax2.set_title('Electric Field Lines – Parallel-Plate Capacitor (κ = 1)',
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig('q1_efield_lines.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q1_efield_lines.png")

# 7.  Plot 3 – |E| along the vertical midline  x = 0
i_mid  = N // 2                       # column index at x ≈ 0
E_mid  = E_mag[:, i_mid]              # field magnitude along x = 0

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(y * 1e3, E_mid, 'b-', lw=2, label='|E(0, y)|')

# Indicate plate positions
ax3.axvline(y_A * 1e3, color='red',   ls='--', lw=1.8,
            label=f'Plate A  y = {y_A*1e3:.2f} mm')
ax3.axvline(y_B * 1e3, color='green', ls='--', lw=1.8,
            label=f'Plate B  y = {y_B*1e3:.2f} mm')

ax3.set_xlabel('y  (mm)', fontsize=12)
ax3.set_ylabel('|E|  (V/m)', fontsize=12)
ax3.set_title('|E| Along Vertical Midline  x = 0  (κ = 1)',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q1_efield_midline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q1_efield_midline.png")
