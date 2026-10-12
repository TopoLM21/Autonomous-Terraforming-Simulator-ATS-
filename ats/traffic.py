"""Swept separation checks; deterministic local detours, no impact dynamics."""
import numpy as np


def closest_separation(a0, a1, b0, b1):
    relative = np.asarray(a0)-b0
    velocity = (np.asarray(a1)-a0)-(np.asarray(b1)-b0)
    norm = float(np.dot(velocity, velocity))
    time = np.clip(-np.dot(relative, velocity)/norm, 0, 1) if norm > 1e-18 else 0
    return float(np.linalg.norm(relative+velocity*time))


def conflicts(drone, old, new, others, starts, separation):
    for other in others:
        if other is drone:
            continue
        if closest_separation(old, new, starts[other.identifier], other.position) < separation-1e-9:
            return other
    return None
