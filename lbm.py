# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "warp-lang>=1.18"]
# ///
"""Sparse lattice-Boltzmann blood-flow solver on the GPU (NVIDIA Warp), D3Q27 (or D3Q19).

Only the fluid cells are stored (the artery fills ~1 % of its bounding box): every cell keeps the index of the
cell it pulls each population from, or a boundary code for that link.

  - lattice: D3Q27 by default. D3Q19 is not rotationally invariant enough at the artery's grid Reynolds numbers
    (u dx / nu ~ 30-60): in a straight pipe it grows spurious secondary vortices (White & Chong 2011, J Comput Phys
    230:6367); D3Q27 doesn't. Set ARTERY_LBM_LATTICE=D3Q19 before importing to compare;
  - collision: regularised BGK (Latt & Chopard 2006: the non-equilibrium part is projected on its second moment,
    which filters the ghost modes) - stable at blood's real viscosity on a 0.1 mm grid (tau ~ 0.505), where the
    two-relaxation-time (TRT) alternative, also here, blows up at the inlet; optional hybrid blend (sigma < 1) with
    the finite-difference stress (Jacob et al. 2018); incompressible equilibrium (He & Luo 1997), so density
    fluctuations don't leak into the momentum; Guo-type body force (for the validation pipes);
  - walls: interpolated bounce-back (Bouzidi et al. 2001, linear), wall distance from the signed distance field,
    so curved walls are placed to second order rather than as a voxel staircase;
  - inlet: bounce-back with a prescribed velocity profile (any orientation), scaled every step by a periodic
    drive waveform (Fourier coefficients evaluated on the GPU);
  - outlets: anti-bounce-back with a prescribed pressure per outlet (any orientation).
Populations are stored as deviations from the rest weights (f - w) for float32 precision. One step = one kernel
(pull streaming + boundaries + collision); runs of steps are captured as CUDA graphs.

Units inside the solver are lattice units (dx = dt = 1, rho0 = 1); `Units` converts. Pressure p = delta_rho / 3.
"""
import os

import numpy as np
import warp as wp

LATTICE = os.environ.get("ARTERY_LBM_LATTICE", "D3Q27")
_C19 = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]]
_CORNERS = [[1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]]
if LATTICE == "D3Q27":
    C = np.array(_C19 + _CORNERS, np.int32)
    W = np.array([8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8)
elif LATTICE == "D3Q19":
    C = np.array(_C19, np.int32)
    W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
else:
    raise ValueError(f"ARTERY_LBM_LATTICE={LATTICE}: use D3Q27 or D3Q19")
Q = len(C)
OPP = np.array([int(np.flatnonzero((C == -c).all(1))[0]) for c in C], np.int32)
# links 1..6 are +x, -x, +y, -y, +z, -z (used by the finite-difference gradient)

WALL, INLET, OUTLET0 = -1, -2, -3  # link codes in the neighbour table (outlet k = OUTLET0 - k)

vecq = wp.types.vector(length=Q, dtype=wp.float32)
vecqi = wp.types.vector(length=Q, dtype=wp.int32)
CX = wp.constant(vecqi(*C[:, 0].tolist()))
CY = wp.constant(vecqi(*C[:, 1].tolist()))
CZ = wp.constant(vecqi(*C[:, 2].tolist()))
WQ = wp.constant(vecq(*W.tolist()))
OP = wp.constant(vecqi(*OPP.tolist()))

wp.set_module_options({"max_unroll": 32})


class Units:
    """Physical <-> lattice units for spacing dx (m), step dt (s), blood density rho (kg/m^3), viscosity nu."""

    def __init__(self, dx, dt, rho, nu):
        self.dx, self.dt, self.rho, self.nu = dx, dt, rho, nu
        self.nu_lu = nu * dt / dx ** 2
        self.tau = 3.0 * self.nu_lu + 0.5
        self.c = dx / dt                              # m/s per lattice velocity unit

    def u(self, u_lu):
        return np.asarray(u_lu) * self.c

    def u_lu(self, u):
        return np.asarray(u) / self.c

    def p(self, p_lu):                                # Pa
        return np.asarray(p_lu) * self.rho * self.c ** 2

    def p_lu(self, p):
        return np.asarray(p) / (self.rho * self.c ** 2)


# ---------------------------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------------------------


@wp.kernel
def k_clock(tick: wp.array(dtype=int), coef: wp.array(dtype=float), nh: int, period: int, ramp: int,
            drive: wp.array(dtype=float)):
    """drive = a0 + sum_k a_k cos(k w t) + b_k sin(k w t), times a smooth start-up ramp; advances the step count."""
    t = tick[0]
    ph = 2.0 * wp.pi * float(t % period) / float(period)
    d = coef[0]
    for k in range(1, nh + 1):
        d += coef[2 * k - 1] * wp.cos(float(k) * ph) + coef[2 * k] * wp.sin(float(k) * ph)
    if t < ramp:
        d *= 0.5 - 0.5 * wp.cos(wp.pi * float(t) / float(ramp))
    drive[0] = d
    tick[0] = t + 1


@wp.func
def _ddx(nbr: wp.array2d(dtype=int), qw: wp.array2d(dtype=float), vel: wp.array(dtype=wp.vec3), n: int,
         i_minus: int, i_plus: int, u0: wp.vec3):
    """du/dx_a at cell n from its two neighbours along axis a (links i_minus: from x - e_a, i_plus: from x + e_a);
    a wall counts as u = 0 at its distance q, an opening as zero gradient (non-uniform central difference)."""
    h1 = float(1.0)
    v1 = u0
    s1 = nbr[i_minus, n]
    if s1 >= 0:
        v1 = vel[s1]
    elif s1 == WALL:
        h1 = wp.max(qw[i_minus, n], 0.1)
        v1 = wp.vec3()
    h2 = float(1.0)
    v2 = u0
    s2 = nbr[i_plus, n]
    if s2 >= 0:
        v2 = vel[s2]
    elif s2 == WALL:
        h2 = wp.max(qw[i_plus, n], 0.1)
        v2 = wp.vec3()
    return (h1 * h1 * (v2 - u0) + h2 * h2 * (u0 - v1)) / (h1 * h2 * (h1 + h2))


@wp.func
def _h3(c: wp.vec3, u: wp.vec3):
    """Third-order equilibrium term / w_i that D3Q27 supports: sum over H_xxy-type and H_xyz Hermite polynomials
    (norms 2 cs^6 and cs^6) times u_x u_x u_y etc. Removes the velocity-cubed error from the off-diagonal stress."""
    t = 1.0 / 3.0
    return (13.5 * ((c[0] * c[0] - t) * (c[1] * u[0] * u[0] * u[1] + c[2] * u[0] * u[0] * u[2])
                    + (c[1] * c[1] - t) * (c[0] * u[0] * u[1] * u[1] + c[2] * u[1] * u[1] * u[2])
                    + (c[2] * c[2] - t) * (c[0] * u[0] * u[2] * u[2] + c[1] * u[1] * u[2] * u[2]))
            + 27.0 * c[0] * c[1] * c[2] * u[0] * u[1] * u[2])


@wp.kernel
def k_step(f_in: wp.array2d(dtype=float), f_out: wp.array2d(dtype=float), nbr: wp.array2d(dtype=int),
           qw: wp.array2d(dtype=float), uin: wp.array(dtype=wp.vec3), inl: wp.array(dtype=int),
           p_out: wp.array(dtype=float), drive: wp.array(dtype=float), force: wp.vec3, om_cell: wp.array(dtype=float), magic: float,
           bouzidi: int, model: int, sigma: float, gal: int, vel_old: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
           rho: wp.array(dtype=float)):
    n = wp.tid()
    # local copies of the lattice constants (referencing the globals inside branches miscompiles in Warp 1.18)
    cx, cy, cz, wq, op = CX, CY, CZ, WQ, OP
    om_p = om_cell[n]                                   # per cell: higher viscosity in the sponge layers
    om_m = 1.0 / (0.5 + magic / (1.0 / om_p - 0.5))
    d = drive[0]
    F = force * d
    up = vel_old[n]
    upsq = wp.dot(up, up)
    f = vecq()
    # --- pull streaming with boundary links
    for i in range(Q):
        src = nbr[i, n]
        io = op[i]
        v = float(0.0)
        if src >= 0:
            v = f_in[i, src]
        elif src == WALL:
            fo = f_in[io, n]
            v = fo
            if bouzidi != 0:
                q = qw[i, n]
                if q < 0.5:
                    m = nbr[io, n]
                    if m >= 0:
                        v = 2.0 * q * fo + (1.0 - 2.0 * q) * f_in[io, m]
                else:
                    v = fo / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * f_in[i, n]
        elif src == INLET:
            uw = uin[n] * d
            cu = float(cx[i]) * uw[0] + float(cy[i]) * uw[1] + float(cz[i]) * uw[2]
            v = f_in[io, n] + 6.0 * wq[i] * cu
        else:
            k = OUTLET0 - src
            cu = float(cx[i]) * up[0] + float(cy[i]) * up[1] + float(cz[i]) * up[2]
            v = -f_in[io, n] + 2.0 * wq[i] * (3.0 * p_out[k] + 4.5 * cu * cu - 1.5 * upsq)
        f[i] = v
    # --- moments (incompressible: u = j, with half the force)
    dr = float(0.0)
    j = wp.vec3()
    for i in range(Q):
        dr += f[i]
        j += wp.vec3(float(cx[i]), float(cy[i]), float(cz[i])) * f[i]
    u = j + F * 0.5
    usq = wp.dot(u, u)
    uF = wp.dot(u, F)
    # Galilean-invariance correction (D3Q27): third-order equilibrium terms, and the remaining diagonal error
    # Ed_a = tau d(u_a^3)/dx_a removed from the stress (finite differences of the previous step's velocities)
    t3 = float(0.0)
    Ed = wp.vec3()
    if gal != 0:
        t3 = 1.0
        gx = _ddx(nbr, qw, vel_old, n, 1, 2, up)
        gy = _ddx(nbr, qw, vel_old, n, 3, 4, up)
        gz = _ddx(nbr, qw, vel_old, n, 5, 6, up)
        tp = 1.0 / om_p
        Ed = wp.vec3(3.0 * tp * u[0] * u[0] * gx[0], 3.0 * tp * u[1] * u[1] * gy[1], 3.0 * tp * u[2] * u[2] * gz[2])
    Dd = -(1.0 - om_p) * Ed
    trD = (Dd[0] + Dd[1] + Dd[2]) / 3.0
    if inl[n] != 0:
        # inlet cells: equilibrium at the prescribed velocity (exact profile; the bounce-back inlet alone lets it
        # drift ~6 % too peaked, which then takes ~15 diameters to develop back)
        ut = uin[n] * d
        utsq = wp.dot(ut, ut)
        for i in range(Q):
            c0 = wp.vec3(float(cx[i]), float(cy[i]), float(cz[i]))
            c0u = wp.dot(c0, ut)
            f_out[i, n] = wq[i] * (dr + 3.0 * c0u + 4.5 * c0u * c0u - 1.5 * utsq + t3 * _h3(c0, ut))
        vel[n] = ut
        rho[n] = dr
        return
    if model == 1:
        # --- regularised BGK: non-equilibrium part projected on its second moment (ghost modes filtered)
        P = wp.mat33()
        for i in range(Q):
            c1 = wp.vec3(float(cx[i]), float(cy[i]), float(cz[i]))
            c1u = wp.dot(c1, u)
            P += wp.outer(c1, c1) * (f[i] - wq[i] * (dr + 3.0 * c1u + 4.5 * c1u * c1u - 1.5 * usq))
        if sigma < 1.0:
            # hybrid (Jacob et al. 2018): blend with the stress from finite-difference velocity gradients of the
            # previous step
            G = wp.matrix_from_rows(_ddx(nbr, qw, vel_old, n, 1, 2, up), _ddx(nbr, qw, vel_old, n, 3, 4, up),
                                    _ddx(nbr, qw, vel_old, n, 5, 6, up))
            P = sigma * P - (1.0 - sigma) / (3.0 * om_p) * (G + wp.transpose(G))
        P = P - wp.diag(Ed)
        trP = (P[0, 0] + P[1, 1] + P[2, 2]) / 3.0
        for i in range(Q):
            c2 = wp.vec3(float(cx[i]), float(cy[i]), float(cz[i]))
            c2u = wp.dot(c2, u)
            c2F = wp.dot(c2, F)
            feq = wq[i] * (dr + 3.0 * c2u + 4.5 * c2u * c2u - 1.5 * usq + t3 * _h3(c2, u))
            # second-order Hermite projection of the non-equilibrium part, + its first moment j - u = -F/2
            fneq = wq[i] * (4.5 * (wp.dot(c2, P * c2) - trP) - 1.5 * c2F)
            src2 = wq[i] * (3.0 * c2F + 9.0 * c2u * c2F - 3.0 * uF)
            f_out[i, n] = feq + (1.0 - om_p) * fneq + (1.0 - 0.5 * om_p) * src2
    else:
        # --- TRT collision with Guo forcing
        for i in range(Q):
            jo = op[i]
            ci = wp.vec3(float(cx[i]), float(cy[i]), float(cz[i]))
            cv = wp.dot(ci, u)
            cF = wp.dot(ci, F)
            eqp = wq[i] * (dr + 4.5 * cv * cv - 1.5 * usq)
            eqm = wq[i] * (3.0 * cv + t3 * _h3(ci, u))
            psi = wq[i] * 4.5 * (ci[0] * ci[0] * Dd[0] + ci[1] * ci[1] * Dd[1] + ci[2] * ci[2] * Dd[2] - trD)
            fp = 0.5 * (f[i] + f[jo])
            fm = 0.5 * (f[i] - f[jo])
            sp = wq[i] * (9.0 * cv * cF - 3.0 * uF)
            sm = wq[i] * 3.0 * cF
            f_out[i, n] = f[i] - om_p * (fp - eqp) - om_m * (fm - eqm) + (1.0 - 0.5 * om_p) * sp + (1.0 - 0.5 * om_m) * sm + psi
    vel[n] = u
    rho[n] = dr


# lattice
# ---------------------------------------------------------------------------------------------------------------


def build_lattice(phi, origin, dx, openings=(), periodic=(), opening_slack=1.0):
    """Sparse lattice from a signed distance on grid nodes (negative = fluid; nodes at origin + (i, j, k) dx).

    openings: dicts {"kind": "inlet" | "outlet", "centre", "normal" (outward), "radius"} in the same units as
    origin and dx; a missing link whose solid node lies beyond an opening's plane, within its radius (+ slack x dx),
    belongs to that opening; every other missing link is a wall. The inlet profile is parabolic in the distance
    from the inlet centre, unit MEAN velocity, pointing inward. periodic: axes (0, 1, 2) that wrap around.
    Returns a dict of numpy arrays (nbr (Q, N) int32, q (Q, N) float32, uin (N, 3), coords (N, 3), ...)."""
    phi = np.asarray(phi, np.float32)
    shape = np.array(phi.shape)
    fluid = phi < 0
    idx = np.full(phi.shape, -1, np.int32)
    coords = np.argwhere(fluid).astype(np.int32)
    N = len(coords)
    idx[tuple(coords.T)] = np.arange(N, dtype=np.int32)
    origin = np.asarray(origin, float)
    nbr = np.empty((Q, N), np.int32)
    qw = np.full((Q, N), 0.5, np.float32)
    uin = np.zeros((N, 3), np.float32)
    phi_x = phi[tuple(coords.T)]
    outlets = [o for o in openings if o["kind"] == "outlet"]
    inlets = [o for o in openings if o["kind"] == "inlet"]
    counts = {"wall": 0, "inlet": 0, "outlet": [0] * len(outlets)}
    for i in range(Q):
        nb = coords - C[i]
        for a in periodic:
            nb[:, a] %= shape[a]
        inside = ((nb >= 0) & (nb < shape)).all(1)
        src = np.full(N, WALL, np.int32)
        src[inside] = idx[tuple(nb[inside].T)]
        miss = np.flatnonzero(src < 0)
        if len(miss):
            ps = origin + nb[miss] * dx
            code = np.full(len(miss), WALL, np.int32)
            for kind, ops in (("inlet", inlets), ("outlet", outlets)):
                for k, o in enumerate(ops):
                    c, nrm = np.asarray(o["centre"], float), np.asarray(o["normal"], float)
                    rel = ps - c
                    ax = rel @ nrm
                    lat = np.linalg.norm(rel - ax[:, None] * nrm, axis=1)
                    hit = (ax >= -1e-6) & (lat < o["radius"] + opening_slack * dx) & (code == WALL)
                    code[hit] = INLET if kind == "inlet" else OUTLET0 - k
            src[miss] = code
            # wall distance fraction along the link (fluid node -> solid node), from the signed distance
            wl = miss[code == WALL]
            if len(wl):
                nbw = nb[wl]
                ok = ((nbw >= 0) & (nbw < shape)).all(1)
                ps_phi = np.full(len(wl), np.nan, np.float32)
                ps_phi[ok] = phi[tuple(nbw[ok].T)]
                q = -phi_x[wl] / (ps_phi - phi_x[wl])
                q = np.where(np.isfinite(q), np.clip(q, 1e-3, 1.0), 0.5)
                qw[i, wl] = q
            counts["wall"] += int((code == WALL).sum())
            counts["inlet"] += int((code == INLET).sum())
            for k in range(len(outlets)):
                counts["outlet"][k] += int((code == OUTLET0 - k).sum())
        nbr[i] = src
    for o in inlets:
        cells = np.flatnonzero((nbr == INLET).any(0))
        c, nrm = np.asarray(o["centre"], float), np.asarray(o["normal"], float)
        rel = origin + coords[cells] * dx - c
        lat = np.linalg.norm(rel - (rel @ nrm)[:, None] * nrm, axis=1)
        prof = np.clip(2.0 * (1.0 - (lat / o["radius"]) ** 2), 0.0, None)
        uin[cells] = (-prof[:, None] * nrm[None]).astype(np.float32)
    return {"nbr": nbr, "q": qw, "uin": uin, "coords": coords, "origin": origin, "dx": dx, "shape": phi.shape,
            "n_outlets": len(outlets), "counts": counts, "openings": inlets + outlets}


def scatter(lat, values, fill=np.nan):
    """Per-cell values -> dense grid (nan outside the fluid)."""
    values = np.asarray(values)
    out = np.full(tuple(lat["shape"]) + values.shape[1:], fill, values.dtype if values.dtype.kind == "f" else float)
    out[tuple(lat["coords"].T)] = values
    return out


# ---------------------------------------------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------------------------------------------


class Solver:
    """Lattice Boltzmann on a sparse lattice. tau from the lattice viscosity; model "trt" (magic = Lambda) or
    "regularised" (BGK with the non-equilibrium part regularised: more robust close to tau = 1/2); sigma < 1 blends
    the regularised stress with the finite-difference one (hybrid regularisation)."""

    def __init__(self, lat, nu_lu, magic=3.0 / 16.0, bouzidi=True, model="trt", sigma=1.0, galilean=None,
                 device=None):
        self.dev = wp.get_device(device)
        self.lat = lat
        self.N = N = lat["nbr"].shape[1]
        self.tau = 3.0 * nu_lu + 0.5
        self.magic = float(magic)
        self.bouzidi = int(bool(bouzidi))
        self.model = {"trt": 0, "regularised": 1}[model]
        self.sigma = float(sigma)
        # Galilean-invariance correction: on by default with D3Q27 (its third-order terms assume D3Q27)
        self.gal = int(LATTICE == "D3Q27" if galilean is None else bool(galilean))
        if self.gal and LATTICE != "D3Q27":
            raise ValueError("the Galilean correction needs D3Q27")
        dev = self.dev
        self.f = [wp.zeros((Q, N), dtype=float, device=dev), wp.zeros((Q, N), dtype=float, device=dev)]
        self.nbr = wp.array(lat["nbr"], dtype=int, device=dev)
        self.q = wp.array(lat["q"], dtype=float, device=dev)
        self.uin = wp.array(lat["uin"], dtype=wp.vec3, device=dev)
        self.inl = wp.array((lat["nbr"] == INLET).any(0).astype(np.int32), dtype=int, device=dev)
        self.p_out = wp.zeros(max(1, lat["n_outlets"]), dtype=float, device=dev)
        self.vels = [wp.zeros(N, dtype=wp.vec3, device=dev), wp.zeros(N, dtype=wp.vec3, device=dev)]
        self.rho = wp.zeros(N, dtype=float, device=dev)
        self.om = wp.full(N, 1.0 / self.tau, dtype=float, device=dev)
        self.tick = wp.zeros(1, dtype=int, device=dev)
        self.drv = wp.zeros(1, dtype=float, device=dev)
        self.force = wp.vec3(0.0, 0.0, 0.0)
        self.set_drive([1.0])
        self.cur = 0
        self.graphs = {}
        self.steps = 0

    # --- inputs
    def set_drive(self, coef, period=1, ramp=0):
        """drive(t) = coef[0] + sum_k coef[2k-1] cos(2 pi k t / period) + coef[2k] sin(...); scales the inlet
        profile (lattice velocity of the mean) and the body force. ramp: smooth start over that many steps."""
        coef = np.asarray(coef, np.float32)
        if len(coef) % 2 == 0:
            coef = np.append(coef, 0.0).astype(np.float32)
        self.coef = wp.array(coef, dtype=float, device=self.dev)
        self.nh, self.period, self.ramp = (len(coef) - 1) // 2, int(period), int(ramp)
        self.graphs = {}

    def set_force(self, F):
        self.force = wp.vec3(*map(float, F))
        self.graphs = {}

    def set_sponge(self, length, tau_max, kinds=("inlet", "outlet")):
        """Raise the viscosity smoothly towards the openings of the given kinds: tau_max at the opening plane, back to
        the fluid's tau at `length` cells inside. Damps the open boundaries, which are unstable at tau ~ 1/2; the
        sponge lies in the straight flow extensions, outside the part of the vessel that is evaluated."""
        lat = self.lat
        P = lat["origin"] + lat["coords"] * lat["dx"]
        tau = np.full(self.N, self.tau)
        for o in lat["openings"]:
            if o["kind"] not in kinds:
                continue
            c, nrm = np.asarray(o["centre"], float), np.asarray(o["normal"], float)
            rel = P - c
            depth = -(rel @ nrm) / lat["dx"]
            side = np.linalg.norm(rel - (rel @ nrm)[:, None] * nrm, axis=1) < 1.5 * o["radius"]
            w = np.clip(1.0 - depth / length, 0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            tau = np.maximum(tau, np.where(side, self.tau + (tau_max - self.tau) * w, self.tau))
        self.tau_cell = tau
        self.om.assign((1.0 / tau).astype(np.float32))

    def set_outlet_pressure(self, p_lu):
        self.p_out.assign(np.asarray(p_lu, np.float32).reshape(-1))

    # --- stepping
    def _launch(self, a):
        wp.launch(k_clock, dim=1, device=self.dev,
                  inputs=[self.tick, self.coef, self.nh, self.period, self.ramp, self.drv])
        wp.launch(k_step, dim=self.N, device=self.dev,
                  inputs=[self.f[a], self.f[1 - a], self.nbr, self.q, self.uin, self.inl, self.p_out, self.drv, self.force,
                          self.om, self.magic, self.bouzidi, self.model, self.sigma, self.gal, self.vels[a], self.vels[1 - a],
                          self.rho])

    def run(self, steps, chunk=200):
        """Advance `steps` steps (CUDA graphs of `chunk` steps; chunk even)."""
        chunk += chunk % 2
        full, rest = divmod(int(steps), chunk)
        if full:
            key = (self.cur, chunk)
            if key not in self.graphs:
                with wp.ScopedCapture(device=self.dev) as cap:
                    a = self.cur
                    for _ in range(chunk):
                        self._launch(a)
                        a = 1 - a
                self.graphs[key] = cap.graph
            for _ in range(full):
                wp.capture_launch(self.graphs[key])
        for _ in range(rest):
            self._launch(self.cur)
            self.cur = 1 - self.cur
        self.steps += int(steps)

    def fields(self):
        """(pressure, velocity) per cell, lattice units, as numpy."""
        return self.rho.numpy() / 3.0, self.vels[self.cur].numpy()

    def ok(self):
        return bool(np.isfinite(self.rho.numpy()).all())
