
"""Two-qubit Hamiltonian and VQE utilities for microscopy-conditioned MXene motifs."""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

def kron2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)

def ry(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

def rz(phi: float) -> np.ndarray:
    return np.array([[np.exp(-1j*phi/2), 0.0],
                     [0.0, np.exp(1j*phi/2)]], dtype=complex)

def cnot_01() -> np.ndarray:
    """CNOT with qubit 0 control and qubit 1 target in |q0 q1> basis."""
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=complex)

def build_hamiltonian(t_parallel: float, t_perp: float, Delta: float) -> np.ndarray:
    """Build H = -t_parallel(XI + IX) - t_perp(XX) + Delta/2(ZI - IZ)."""
    return (
        -t_parallel * (kron2(X, I2) + kron2(I2, X))
        -t_perp * kron2(X, X)
        +0.5 * Delta * (kron2(Z, I2) - kron2(I2, Z))
    )

def exact_ground_energy(H: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(H)[0].real)

def simple_ansatz_state(theta: float) -> np.ndarray:
    """|psi(theta)> = [Ry(theta) ⊗ Ry(theta)] |00>."""
    psi0 = np.array([1,0,0,0], dtype=complex)
    U = kron2(ry(theta), ry(theta))
    return U @ psi0

def enhanced_ansatz_state(params: np.ndarray) -> np.ndarray:
    """Two-layer entangled ansatz with 8 parameters:
    [phi0_1, theta0_1, phi1_1, theta1_1, phi0_2, theta0_2, phi1_2, theta1_2].
    """
    if len(params) != 8:
        raise ValueError("Enhanced ansatz requires exactly 8 parameters.")
    phi0_1, th0_1, phi1_1, th1_1, phi0_2, th0_2, phi1_2, th1_2 = params
    psi0 = np.array([1,0,0,0], dtype=complex)
    U1 = kron2(ry(th0_1) @ rz(phi0_1), ry(th1_1) @ rz(phi1_1))
    U2 = kron2(ry(th0_2) @ rz(phi0_2), ry(th1_2) @ rz(phi1_2))
    return U2 @ cnot_01() @ U1 @ psi0

def energy(psi: np.ndarray, H: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, H @ psi)))

def run_vqe_simple(H: np.ndarray, theta0: float = 0.1, maxiter: int = 300) -> dict:
    def obj(x):
        return energy(simple_ansatz_state(float(x[0])), H)
    res = minimize(obj, x0=np.array([theta0]), method="COBYLA", options={"maxiter": maxiter})
    return {"E_vqe": obj(res.x), "theta_opt": res.x, "success": bool(res.success)}

def run_vqe_enhanced(H: np.ndarray, seed: int = 0, maxiter: int = 800) -> dict:
    rng = np.random.default_rng(seed)
    x0 = 0.05 * rng.normal(size=8)
    def obj(x):
        return energy(enhanced_ansatz_state(np.asarray(x)), H)
    res = minimize(obj, x0=x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
    return {"E_vqe": obj(res.x), "theta_opt": res.x, "success": bool(res.success)}
