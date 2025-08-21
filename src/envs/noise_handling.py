import numpy as np


def random_waypoint_3d(P0, P1, fraction, max_radius_frac=0.2, radial_mode="uniform"):
    """
    Insert a random waypoint between P0 and P1, offset within a disk
    perpendicular to the segment at a random along-path fraction t.    Args:


        P0, P1: (3,) arrays
        max_radius_frac: disk radius as a fraction of segment length
        radial_mode: "uniform" (area-uniform in disk) or "gaussian" (isotropic)
    """
    P0 = np.asarray(P0, dtype=float)
    P1 = np.asarray(P1, dtype=float)

    seg = P1 - P0
    L = np.linalg.norm(seg)
    if L == 0:
        raise ValueError("P0 and P1 must be different points.")
    d = seg / L  # unit direction

    # 1) choose along-path fraction
    t = fraction
    base = (1 - t) * P0 + t * P1

    # 2) build an orthonormal basis (u, v) for the plane perpendicular to d
    # pick a vector not parallel to d (use the axis with smallest |component|)
    k = np.argmin(np.abs(d))
    a = np.zeros(3); a[k] = 1.0
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)  # already unit if d and u are unit and orthogonal

    # 3) sample an offset in that plane
    Rmax = max_radius_frac * L
    if radial_mode == "uniform":
        # area-uniform in disk: r ~ Rmax * sqrt(U), theta ~ U[0, 2π)
        r = Rmax * np.sqrt(np.random.uniform())
        theta = 2 * np.pi * np.random.uniform()
        offset = r * (np.cos(theta) * u + np.sin(theta) * v)
    elif radial_mode == "gaussian":
        # isotropic 2D gaussian in the plane, then cap by ~3σ to avoid huge jumps
        sigma = Rmax / 3.0
        z1, z2 = np.random.normal(size=2)
        offset = (z1 * u + z2 * v) * sigma
    else:
        raise ValueError("radial_mode must be 'uniform' or 'gaussian'.")

    return base + offset, 

#