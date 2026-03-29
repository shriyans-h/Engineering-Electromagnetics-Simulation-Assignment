import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1.  Grid & boundary parameters
N        = 100          # 100 × 100 grid
V_ground = 0.0          # bottom row potential
V_cloud  = 100.0        # top    row potential
max_iter = 20000        # maximum iterations (convergence usually within ~10000)
tol      = 1e-4         # convergence tolerance (max change per iteration)

SIDE_WALL_LINEAR = False  # True → side walls interpolated; False → free

# Needle definition
i_needle = N // 2          # centre column (x = 50th column for N=100)
j_needle_base = 0          # needle starts at ground row j=0
j_needle_tip  = N // 2     # needle extends up to the midpoint row

# 2.  Initialise V
jj = np.arange(N)
V  = np.outer(jj / (N - 1), np.ones(N)) * (V_cloud - V_ground) + V_ground
# shape: V[j, i], j=0 → bottom (ground), j=N-1 → top (cloud)

# 3.  Boundary-condition helper
def apply_bc(V):
    """Enforce all fixed-potential boundaries."""
    # Top and bottom (cloud & ground)
    V[0,  :] = V_ground   # j=0:   bottom row (ground)
    V[-1, :] = V_cloud    # j=N-1: top    row (cloud)

    # Needle: entire column i_needle from row 0 up to row j_needle_tip
    V[j_needle_base : j_needle_tip + 1, i_needle] = V_ground

    # Side walls (optional)
    if SIDE_WALL_LINEAR:
        lin = np.linspace(V_ground, V_cloud, N)
        V[:, 0]  = lin   # left  wall
        V[:, -1] = lin   # right wall

# 4.  Iterative solver (Gauss-Seidel style, vectorised)
apply_bc(V)

print("Solving Laplace's equation for lightning rod …")
for iteration in range(1, max_iter + 1):
    V_old = V.copy()

    # Nearest-neighbour averaging for all interior nodes
    V[1:-1, 1:-1] = 0.25 * (
        V[2:,   1:-1] +   # north  (j+1)
        V[:-2,  1:-1] +   # south  (j-1)
        V[1:-1,  2:] +    # east   (i+1)
        V[1:-1, :-2]      # west   (i-1)
    )
    apply_bc(V)   # re-enforce BCs

    # Check convergence every 500 iterations
    if iteration % 500 == 0:
        max_diff = np.max(np.abs(V - V_old))
        print(f"  iter {iteration:6d}   max ΔV = {max_diff:.2e}")
        if max_diff < tol:
            print(f"  Converged at iteration {iteration}.")
            break

print("Solver finished.")

# 5.  Electric field  E = −∇V
# Grid spacing = 1 unit  (indices used directly)
dVdy, dVdx = np.gradient(V, 1.0, 1.0)   # axis-0 = j(y), axis-1 = i(x)
Ex = -dVdx
Ey = -dVdy
E_mag = np.sqrt(Ex**2 + Ey**2)

# Coordinate arrays (in grid units)
i_arr = np.arange(N)   # x  (column index)
j_arr = np.arange(N)   # y  (row index, 0 = ground)
IX, JY = np.meshgrid(i_arr, j_arr)

# 6.  Plot 1 – Heat map of electric potential V
fig1, ax1 = plt.subplots(figsize=(7, 7))
im1 = ax1.imshow(V, origin='lower', cmap='plasma',
                 vmin=V_ground, vmax=V_cloud, aspect='equal')
plt.colorbar(im1, ax=ax1, label='Potential  V  (V)', fraction=0.046)

# Draw needle as a white line
ax1.plot([i_needle, i_needle],
         [j_needle_base, j_needle_tip], 'w-', lw=2.5, label='Needle (V=0)')
ax1.plot(i_needle, j_needle_tip, 'w^', ms=8)   # mark the tip

ax1.set_xlabel('x  (grid index)', fontsize=11)
ax1.set_ylabel('y  (grid index, 0=ground)', fontsize=11)
ax1.set_title('Potential Heat Map – Lightning Rod Simulation',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('q3_potential_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_potential_heatmap.png")

# 7.  Plot 2 – V heat map with E-field vector overlay
step = 5   # subsample every 5 grid points for quiver
IXq = IX[::step, ::step]
JYq = JY[::step, ::step]
EXq = Ex[::step, ::step]
EYq = Ey[::step, ::step]
Enorm = np.hypot(EXq, EYq) + 1e-12    # normalise for uniform arrow length

fig2, ax2 = plt.subplots(figsize=(8, 8))
im2 = ax2.imshow(V, origin='lower', cmap='plasma',
                 vmin=V_ground, vmax=V_cloud, aspect='equal', alpha=0.85)
plt.colorbar(im2, ax=ax2, label='Potential  V  (V)', fraction=0.046)

# Unit arrows coloured by field magnitude
mag_q = np.hypot(EXq, EYq)
ax2.quiver(IXq, JYq, EXq/Enorm, EYq/Enorm,
           mag_q, cmap='cool', scale=35, width=0.004, alpha=0.85)

# Needle
ax2.plot([i_needle, i_needle],
         [j_needle_base, j_needle_tip], 'w-', lw=3, label='Needle')
ax2.plot(i_needle, j_needle_tip, 'w^', ms=9)

ax2.set_xlabel('x  (grid index)', fontsize=11)
ax2.set_ylabel('y  (grid index)', fontsize=11)
ax2.set_title('E = −∇V  Vectors Overlaid on Potential Map',
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('q3_efield_overlay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_efield_overlay.png")

# 8.  Plot 3 – |E| magnitude heat map (highlights the tip enhancement)
fig3, ax3 = plt.subplots(figsize=(7, 7))
# Log scale brings out both weak and strong regions
E_log = np.log1p(E_mag)
im3   = ax3.imshow(E_log, origin='lower', cmap='hot', aspect='equal')
plt.colorbar(im3, ax=ax3, label='log(1 + |E|)  (log-V/unit)', fraction=0.046)

ax3.plot([i_needle, i_needle],
         [j_needle_base, j_needle_tip], 'c-', lw=2.5, label='Needle')
ax3.plot(i_needle, j_needle_tip, 'c^', ms=9)

ax3.set_xlabel('x  (grid index)', fontsize=11)
ax3.set_ylabel('y  (grid index)', fontsize=11)
ax3.set_title('|E| Magnitude Map – Field Enhancement at Needle Tip',
              fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('q3_emag_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_emag_heatmap.png")

# 9.  Plot 4 – |E| along the vertical centre line  (x = i_needle)
#              Quantifies the tip-enhancement effect
E_centre = E_mag[:, i_needle]    # |E| at x = centre, for all y

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(j_arr, E_centre, 'r-', lw=2, label='|E|  along centre line')
ax4.axvline(j_needle_tip, color='blue', ls='--', lw=1.8,
            label=f'Needle tip  y = {j_needle_tip}')
ax4.axvline(j_needle_base, color='green', ls=':', lw=1.5,
            label=f'Needle base y = {j_needle_base}')

# Annotate maximum (should be at or just above the tip)
j_peak = int(np.argmax(E_centre))
ax4.annotate(f'Peak  |E| = {E_centre[j_peak]:.2f}\nat y = {j_peak}',
             xy=(j_peak, E_centre[j_peak]),
             xytext=(j_peak + 5, E_centre[j_peak] * 0.85),
             arrowprops=dict(arrowstyle='->', color='k'),
             fontsize=10, color='darkred')

ax4.set_xlabel('y  (grid index, 0=ground)', fontsize=11)
ax4.set_ylabel('|E|  (V/unit)', fontsize=11)
ax4.set_title('Field Enhancement Along Vertical Centre Line  (x = centre)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q3_efield_centreline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_efield_centreline.png")

# 10.  Print quantitative summary
E_far_from_tip = E_centre[N - 2]          # field near the cloud (away from needle)
E_at_tip       = E_centre[j_needle_tip]   # field just at the needle tip row
print("\n─── Field Enhancement Summary ───")
print(f"  |E| near cloud top   (y≈{N-2:d}):       {E_far_from_tip:.3f} V/unit")
print(f"  |E| at needle tip    (y={j_needle_tip:d}):    {E_at_tip:.3f} V/unit")
if E_far_from_tip > 0:
    print(f"  Enhancement factor (tip / far-field): {E_at_tip/E_far_from_tip:.1f}×")
