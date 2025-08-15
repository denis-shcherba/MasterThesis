import numpy as np

class GaussianNoiseAdder:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
        self.previous_noise = None

    def add_noise(self, data, num_states_to_perturb=2):
        # Ensure data is a NumPy array
        data = np.asarray(data)
        
        # Check if the data has enough elements to perturb
        if data.size < num_states_to_perturb:
            raise ValueError(f"Data has size {data.size} but needs at least {num_states_to_perturb} elements to perturb.")

        # Generate fresh Gaussian noise for the specified number of states
        noise = np.random.normal(self.mean, self.std, size=num_states_to_perturb)
        
        # Accumulate with previous noise (if any)
        if self.previous_noise is not None:
            # Check if previous noise shape matches the current one
            if self.previous_noise.shape[0] != num_states_to_perturb:
                # Reset if the number of states to perturb changes
                print("Warning: Number of states to perturb changed. Resetting accumulated noise.")
                self.previous_noise = None
            else:
                noise += self.previous_noise

        # Save current noise for the next time
        self.previous_noise = noise

        # Create a copy of the data to avoid modifying the original array
        noisy_data = data.copy()
        
        # Add the 2D noise to the first 'num_states_to_perturb' elements of the data
        noisy_data[:num_states_to_perturb] += noise

        return noisy_data

def random_waypoint_3d(P0, P1, max_radius_frac=0.2, radial_mode="uniform"):
    """
    Insert a random waypoint between P0 and P1, offset within a disk
    perpendicular to the segment at a random along-path fraction t.

    Args:
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
    t = np.random.uniform(0.0, 1.0)
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

    return base + offset, t

# ---- example usage ----
P0 = np.array([0.0, 0.0, 0.0])
P1 = np.array([3.0, 4.0, 5.0])

waypoint, t = random_waypoint_3d(P0, P1, max_radius_frac=0.25, radial_mode="uniform")
print(f"t={t:.3f}, waypoint={waypoint}")