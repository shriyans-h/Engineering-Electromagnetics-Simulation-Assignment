"""
  TO CHANGE CHARGE MAGNITUDE:   edit  Q_norm  below
  TO CHANGE CHARGE POSITION  :  edit  d       below
"""

import numpy as np
import matplotlib.pyplot as plt

# 1.  Parameters
R      = 2.0     # sphere radius  [simulation units]
d      = 4.0     # charge distance from origin  [simulation units]
Q_norm = 1.0     # normalised charge strength K = λ/(2πε₀);  scale for Q = 10 µC

# ── Change Q_norm or d here to explore different scenarios ──────────────────
# Q_norm = 2.0    # doubled charge
# d      = 5.0    # charge moved farther away

# Image charge position
x_img  = R**2 / d   # = 1.0  for default parameters

# 2.  Build grid
N = 400
xg = np.linspace(-6, 6, N)
yg = np.linspace(-6, 6, N)
X, Y = np.meshgrid(xg, yg)


# 3.  Compute potential  V(x, y)  using method of images
eps = 1e-6   # small offset to avoid log(0) at the charge location

r_real  = np.sqrt((X - d    )**2 + Y**2) + eps   # dist to real charge
r_image = np.sqrt((X - x_img)**2 + Y**2) + eps   # dist to image charge

# V = K·ln(d·r_image / (R·r_real))  ← zero on the cylinder, see docstring
V = Q_norm * np.log((d * r_image) / (R * r_real))

# Mask sphere interior (not physically meaningful)
sphere_mask = (X**2 + Y**2) <= R**2
V[sphere_mask] = np.nan

# Clip extreme values near the point charge for clean contour plots
V_clip = np.clip(V, -6, 6)

# 4.  Electric field  E = −∇V  (numerical gradient on the grid)
dg = xg[1] - xg[0]   # uniform grid spacing
dVdy, dVdx = np.gradient(np.nan_to_num(V, nan=0.0), dg, dg)
Ex = -dVdx
Ey = -dVdy
# Zero out field inside sphere
Ex[sphere_mask] = 0.0
Ey[sphere_mask] = 0.0
E_mag = np.sqrt(Ex**2 + Ey**2)

# 5.  Induced surface charge density  σ(θ)   (exact formula for 2D cylinder)
theta = np.linspace(0, 2*np.pi, 720)
# σ(θ) = −(λ/2πR) · (d²−R²)/(d²+R²−2dR cosθ)
sigma = -(Q_norm / (2 * np.pi * R)) * (d**2 - R**2) / \
        (d**2 + R**2 - 2*d*R*np.cos(theta))

# 6.  Plot 1 – Induced surface charge density
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

# (a) σ vs θ  (polar plot of magnitude)
ax_polar = fig1.add_subplot(121, projection='polar')
ax_polar.plot(theta, np.abs(sigma), 'b-', lw=2)
ax_polar.fill(theta, np.abs(sigma), alpha=0.25)
ax_polar.set_title('|σ(θ)|  on sphere surface\n(polar)', fontweight='bold', pad=15)
ax_polar.set_xlabel('θ  (rad)')

# (b) σ vs θ  (Cartesian – signed, so direction visible)
ax_cart = fig1.add_subplot(122)
ax_cart.plot(np.degrees(theta), sigma, 'r-', lw=2)
ax_cart.axhline(0, color='k', lw=0.8, ls='--')
ax_cart.set_xlabel('θ  (degrees)', fontsize=11)
ax_cart.set_ylabel('σ(θ)  [normalised  C/m]', fontsize=11)
ax_cart.set_title('Induced Surface Charge Density', fontweight='bold')
ax_cart.set_xlim(0, 360)
ax_cart.set_xticks([0, 90, 180, 270, 360])
ax_cart.grid(True, alpha=0.4)

# Remove the duplicate axes1 created by subplots
fig1.delaxes(axes1[0])
fig1.delaxes(axes1[1])

fig1.suptitle(f'Q2 – Grounded Sphere  (R={R}, d={d}, Q_norm={Q_norm})',
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_induced_charge.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2_induced_charge.png")

# 7.  Plot 2 – Equipotential contour map
fig2, ax2 = plt.subplots(figsize=(8, 8))

levels = np.linspace(-5, 5, 60)
cf = ax2.contourf(X, Y, V_clip, levels=levels, cmap='RdBu_r', extend='both')
cs = ax2.contour( X, Y, V_clip, levels=20, colors='k', linewidths=0.4)
plt.colorbar(cf, ax=ax2, label='Potential  V  (normalised units)', fraction=0.046)

# Draw sphere boundary
theta_c = np.linspace(0, 2*np.pi, 360)
ax2.fill(R*np.cos(theta_c), R*np.sin(theta_c), color='grey', alpha=0.6,
         label=f'Sphere  (R={R}, V=0)')
ax2.plot(R*np.cos(theta_c), R*np.sin(theta_c), 'k-', lw=1.5)

# Mark the point charge
ax2.plot(d, 0, 'r*', ms=14, label=f'Point charge  Q_norm={Q_norm}  at ({d},0)')
ax2.plot(x_img, 0, 'b^', ms=10, label=f'Image charge at ({x_img},0)')

ax2.set_xlim(-6, 6); ax2.set_ylim(-6, 6)
ax2.set_xlabel('x', fontsize=12); ax2.set_ylabel('y', fontsize=12)
ax2.set_title(f'Equipotential Contours – Point Charge Near Grounded Sphere',
              fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig('q2_equipotential.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2_equipotential.png")

# 8.  Plot 3 – Electric field vector (quiver) plot
# Subsample for quiver readability
step = 16
Xq, Yq = X[::step, ::step], Y[::step, ::step]
EXq, EYq = Ex[::step, ::step], Ey[::step, ::step]
EMq = E_mag[::step, ::step]

fig3, ax3 = plt.subplots(figsize=(8, 8))
cf3 = ax3.contourf(X, Y, np.log1p(E_mag), levels=50, cmap='hot_r', alpha=0.7)
plt.colorbar(cf3, ax=ax3, label='log(1 + |E|)', fraction=0.046)

# Normalise quiver arrows so all are same length (direction only)
EQ_norm = np.hypot(EXq, EYq) + 1e-12
ax3.quiver(Xq, Yq, EXq/EQ_norm, EYq/EQ_norm,
           color='white', scale=40, width=0.003, alpha=0.9)

# Sphere
ax3.fill(R*np.cos(theta_c), R*np.sin(theta_c), color='grey', alpha=0.9)
ax3.plot(R*np.cos(theta_c), R*np.sin(theta_c), 'k-', lw=1.5)
ax3.plot(d, 0, 'c*', ms=14, label=f'Charge at ({d},0)')

ax3.set_xlim(-6, 6); ax3.set_ylim(-6, 6)
ax3.set_xlabel('x', fontsize=12); ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Electric Field Distribution – Point Charge Near Grounded Sphere',
              fontweight='bold')
ax3.legend(fontsize=9); ax3.set_aspect('equal')
plt.tight_layout()
plt.savefig('q2_efield_vectors.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2_efield_vectors.png")
