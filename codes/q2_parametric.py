import numpy as np
import matplotlib.pyplot as plt

# 1.  Core functions (method of images for 2D grounded cylinder)
R = 2.0   # sphere radius

def compute_V(Q_norm, d, N=300):
    """Return V on N×N grid  [-6,6]² for given charge Q_norm at (d,0)."""
    x = np.linspace(-6, 6, N)
    y = np.linspace(-6, 6, N)
    X, Y = np.meshgrid(x, y)

    x_img  = R**2 / d
    eps    = 1e-6
    r_real  = np.sqrt((X - d    )**2 + Y**2) + eps
    r_image = np.sqrt((X - x_img)**2 + Y**2) + eps

    V = Q_norm * np.log((d * r_image) / (R * r_real))
    V[(X**2 + Y**2) <= R**2] = np.nan   # mask sphere interior
    return np.clip(V, -8, 8), X, Y

def sigma_theta(Q_norm, d):
    """Induced surface charge density as function of angle θ."""
    theta = np.linspace(0, 2*np.pi, 720)
    s = -(Q_norm / (2*np.pi*R)) * (d**2 - R**2) / \
        (d**2 + R**2 - 2*d*R*np.cos(theta))
    return theta, s

def efield_xaxis(Q_norm, d):
    """
    Analytical |E| along the x-axis (y=0) outside the sphere for x > R.
    Computed from the potential derivative.
    """
    xv = np.linspace(R + 0.05, 6, 300)
    x_img = R**2 / d
    eps   = 1e-9
    # dV/dx evaluated at y=0:
    # V = K·[ln(d) + ln(r_img) - ln(R) - ln(r_real)]
    # dV/dx = K·[(x-x_img)/r_img² - (x-d)/r_real²]
    r_real  = np.abs(xv - d)   + eps
    r_image = np.abs(xv - x_img) + eps
    dVdx = Q_norm * ((xv - x_img)/r_image**2 - (xv - d)/r_real**2)
    Ex   = -dVdx
    return xv, np.abs(Ex)

# 2.  Case A – Vary charge magnitude  (d = 4 fixed)
d_fixed     = 4.0
Q_values    = [0.5, 1.0, 2.0, 3.0]   # change / add values as needed
colors_Q    = ['steelblue', 'green', 'orange', 'red']

# ── 2A.i  σ(θ) for different Q ───────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
for Q_, c_ in zip(Q_values, colors_Q):
    th, sig = sigma_theta(Q_, d_fixed)
    ax1.plot(np.degrees(th), sig, color=c_, lw=2, label=f'Q_norm = {Q_}')
ax1.axhline(0, color='k', lw=0.8, ls='--')
ax1.set_xlabel('θ (degrees)', fontsize=11)
ax1.set_ylabel('σ(θ)  [normalised]', fontsize=11)
ax1.set_title(f'Induced σ(θ) – Varying Q_norm  (d = {d_fixed})',
              fontweight='bold')
ax1.set_xlim(0, 360); ax1.set_xticks([0, 90, 180, 270, 360])
ax1.legend(); ax1.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q2p_sigma_varyQ.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2p_sigma_varyQ.png")

# ── 2A.ii  Equipotential contours for different Q (2×2 grid) ────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 13))
theta_c = np.linspace(0, 2*np.pi, 360)
for idx, (Q_, ax_) in enumerate(zip(Q_values, axes2.flat)):
    V_, Xg, Yg = compute_V(Q_, d_fixed)
    lv = np.linspace(-5*Q_, 5*Q_, 50)
    cf = ax_.contourf(Xg, Yg, V_, levels=lv, cmap='RdBu_r', extend='both')
    ax_.contour(Xg, Yg, V_, levels=12, colors='k', linewidths=0.4)
    plt.colorbar(cf, ax=ax_, fraction=0.046)
    ax_.fill(R*np.cos(theta_c), R*np.sin(theta_c), color='grey', alpha=0.7)
    ax_.plot(d_fixed, 0, 'r*', ms=12)
    ax_.set_xlim(-6, 6); ax_.set_ylim(-6, 6)
    ax_.set_aspect('equal')
    ax_.set_title(f'Q_norm = {Q_}', fontweight='bold')
    ax_.set_xlabel('x'); ax_.set_ylabel('y')
fig2.suptitle(f'Equipotentials – Varying Q_norm  (d={d_fixed})',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q2p_equi_varyQ.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2p_equi_varyQ.png")

# 3.  Case B – Vary distance  (Q_norm = 1 fixed)
Q_fixed   = 1.0
d_values  = [2.5, 4.0, 5.0, 5.8]   # must be > R=2; change as needed
colors_d  = ['steelblue', 'green', 'orange', 'red']

# ── 3B.i  σ(θ) for different d ───────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))
for d_, c_ in zip(d_values, colors_d):
    th, sig = sigma_theta(Q_fixed, d_)
    ax3.plot(np.degrees(th), sig, color=c_, lw=2, label=f'd = {d_}')
ax3.axhline(0, color='k', lw=0.8, ls='--')
ax3.set_xlabel('θ (degrees)', fontsize=11)
ax3.set_ylabel('σ(θ)  [normalised]', fontsize=11)
ax3.set_title(f'Induced σ(θ) – Varying Distance d  (Q_norm = {Q_fixed})',
              fontweight='bold')
ax3.set_xlim(0, 360); ax3.set_xticks([0, 90, 180, 270, 360])
ax3.legend(); ax3.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q2p_sigma_varyd.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2p_sigma_varyd.png")

# ── 3B.ii  |E| along x-axis for different d ─────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
for d_, c_ in zip(d_values, colors_d):
    xv, E_ = efield_xaxis(Q_fixed, d_)
    ax4.plot(xv, E_, color=c_, lw=2, label=f'd = {d_}')
    ax4.axvline(d_, color=c_, lw=0.8, ls=':')   # mark charge position
ax4.axvline(R, color='k', lw=1.5, ls='--', label=f'Sphere surface  R={R}')
ax4.set_xlabel('x  (along x-axis, y=0)', fontsize=11)
ax4.set_ylabel('|E|  (normalised)', fontsize=11)
ax4.set_title(f'|E| Along x-Axis – Varying d  (Q_norm = {Q_fixed})',
              fontweight='bold')
ax4.set_xlim(R, 6); ax4.legend(); ax4.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q2p_efield_varyd.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2p_efield_varyd.png")

# ── 3B.iii  Equipotential contours for different d (2×2 grid) ───────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 13))
for d_, ax_ in zip(d_values, axes5.flat):
    V_, Xg, Yg = compute_V(Q_fixed, d_)
    cf = ax_.contourf(Xg, Yg, V_, levels=50, cmap='RdBu_r', extend='both')
    ax_.contour(Xg, Yg, V_, levels=15, colors='k', linewidths=0.4)
    plt.colorbar(cf, ax=ax_, fraction=0.046)
    ax_.fill(R*np.cos(theta_c), R*np.sin(theta_c), color='grey', alpha=0.7)
    ax_.plot(d_, 0, 'r*', ms=12)
    ax_.set_xlim(-6, 6); ax_.set_ylim(-6, 6); ax_.set_aspect('equal')
    ax_.set_title(f'd = {d_}', fontweight='bold')
    ax_.set_xlabel('x'); ax_.set_ylabel('y')
fig5.suptitle(f'Equipotentials – Varying d  (Q_norm={Q_fixed})',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q2p_equi_varyd.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q2p_equi_varyd.png")
