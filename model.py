"""
Forked from ../code/model.py for the trigger-suppression-policy experiment
(a standalone comparison, not part of the calibrated paper pipeline in
../code, which this fork does not modify). Only change from the original:
the transmission-rate and cost dispatch blocks are refactored so `oracle`
(which signal drives the scenario -- true phi_i or reported iota_hat_i) and
`policy_fn` (how that signal maps to a multiplier) are independent axes
instead of mutually exclusive branches, so a policy_fn can be evaluated
against the true, noise-free phi_i signal as well as the reported one; and
a new `cum_active_time` accumulator tracks population-weighted node-time
under any suppression (mult_now < 1), for policies -- like a threshold
trigger -- where "how long was suppression on" is a meaningful quantity in
its own right, distinct from cum_suppression's effect-weighted effort.
Every other line, including all docstring content below describing the
model's dynamics, is unchanged from ../code/model.py.

Core stochastic simulation of the CB-EWS closed-loop model specified in
cbews_model_v2.tex, Section 2 (Model construction).

State per node: X_SN, X_SI, X_AI (S is implied: S = N - X_SN - X_SI - X_AI).
Time is nondimensional, one unit = the mean SI infectious period 1/r;
dt = 1/150 reproduces the paper's "one step = one day" convention
(Section 5.1, Implementation) under the same 1/r ~= 150 days figure used
elsewhere in the paper.

Delays implemented (Section 2.3, eqs. XSNdim/XAIdim -- nondimensional forms
eq. xSN/eq. xAI; Section 2.4, eq. xAIhat):
  - new SN cases at time t reflect exposure at t - Tdel.
  - new AI cases at time t reflect exposure at t - Tdel - 1/theta: the
    asymptomatic pathway has no symptomatic stage to mark the end of the
    incubation period, but still needs the same gametocyte-maturation wait
    1/theta (rate theta = mu_g/r) before becoming infectious.
  - the AI nowcast (eq. xAIhat) is driven by the SN threat estimate lagged
    by that same 1/theta.

Two "reported threat" quantities, kept as two separate variables:
  - iota_win_hat = hat_iota_{i,k} (eq. iotahat): the direct per-window
    incidence-rate estimate, held constant for the whole window, driving
    only the nowcast ODE (eqs. xSNhat/xAIhat).
  - iota_hat = hat_iota_i(t) := hat_phi_i(t) (eq. pool): the continuously
    spatially-pooled nowcast prevalence, recomputed every step, driving the
    transmission-rate exponent and the suppression cost.

The nowcast state Xh_SN/Xh_SI/Xh_AI is not itself required to stay
non-negative: it is a linear filter of hat_iota_{i,k} (eq. iotahat), and
nothing else in the model reads it directly except the spatial pooling
below. Non-negativity and reporting-noise suppression are both enforced
exactly once, on the pooled result (eq. nowcastclip):
hat_iota_i(t) := max(0, hat_phi_i(t) - c_i), since pooling first averages
several nodes' independent estimates together, which shrinks the spread of
the quantity being clipped and, with it, the bias the clip itself
introduces. c_i (eq. threshold) is a per-node reporting-noise threshold set
from N_j and W_ij (row i of the spatial-coupling matrix): a node whose
mixing weight is spread across many neighbours needs a smaller threshold
than one concentrated on few, since it already benefits more from pooling's
own noise reduction.

Spatial coupling: W = (1-eps) I + eps M (eq. W). This module does NOT
construct M -- its exact form (distance kernel, gravity/radiation model,
ward adjacency, ...) has not been decided yet. `simulate` takes M as a
required, sparse, row-stochastic matrix; `_placeholder_ring_M` below exists
only to smoke-test this file and must not be used for real results.

phi* (the public-facing threat-level threshold) does not appear here: per
Section 2.4, translating hat_iota_i into a discrete public threat level is a
presentation-layer policy choice, not part of the model's dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------
@dataclass
class Params:
    """The model's parameters (Table 1) minus phi*, plus numerics.
    theta, gamma, Tdel, alpha, rho, nu0 are the nondimensional composites
    of eq. composite; R0, sigma, Delta are already dimensionless."""
    R0: float = 2.0          # mean basic reproduction number (bar R0 if
                              # seasonal_delta != 0, eq. R0seasonal)
    sigma: float = 0.3       # fraction symptomatic
    theta: float = 2.0       # SN -> SI maturation rate (= mu_g / r)
    gamma: float = 0.5       # AI clearance rate (= r_a / r)
    Tdel: float = 0.1        # incubation delay (= r * tau)
    alpha: float = 1.0       # behavioural efficacy
    penalty: str = "exp"     # suppression-cost functional form (eq. costZ
                              # generalises to three forms of N*f(alpha*Z)*dt:
                              # "exp" = exp(u)-1 (the paper default), "linear"
                              # = u, "quadratic" = u^2/2, u := alpha*Z. Prices
                              # the same driving signal Z and reuses the same
                              # alpha throughout; does not affect the
                              # suppression multiplier exp(-alpha*Z) itself,
                              # only how that same effort is costed.
    rho: float = 0.3         # reporting probability
    nu0: float = 0.01        # background false-report rate, per capita
    Delta: float = 0.05      # reporting window length (nondimensional)
    dt: float = 1.0 / 150    # step size; 1/150 ~= one day (1/r ~= 150 d)
    c_min: int = 0           # small-count suppression threshold (Remark, Sec 2.4)
    kappa: float = 0.3       # reporting-noise threshold multiplier (eq. threshold)
    seasonal_delta: float = 0.0    # seasonal forcing amplitude (eq. R0seasonal);
                                    # 0 = constant R0, the pre-seasonal behaviour
    seasonal_t_peak: float = 0.0   # nondimensional time of peak transmission
                                    # within one cycle
    seasonal_period: float = 1.0   # nondimensional length of one seasonal cycle
                                    # (e.g. 365 days / (1/r) in days); irrelevant
                                    # when seasonal_delta = 0
    seasonal_year_multipliers: "np.ndarray | None" = None   # per-year multiplier
        # on seasonal_delta, indexed by floor(t / seasonal_period) and wrapped;
        # None = every year identical (the regular cycle). For illustrating
        # irregular, not strictly periodic, forcing -- fixed and reproducible
        # for a given array, not resampled inside simulate().


def _R0_of_t(p: "Params", t_nondim):
    """R0(t) (eq. R0seasonal): constant at p.R0 when seasonal_delta = 0,
    otherwise a cosine of amplitude seasonal_delta (scaled by that year's
    entry in seasonal_year_multipliers, if given) around that mean, peaking
    at seasonal_t_peak every seasonal_period."""
    if p.seasonal_delta == 0.0:
        return p.R0
    delta = p.seasonal_delta
    if p.seasonal_year_multipliers is not None:
        year = int(t_nondim // p.seasonal_period) % len(p.seasonal_year_multipliers)
        delta = delta * p.seasonal_year_multipliers[year]
    phase = 2 * np.pi * (t_nondim - p.seasonal_t_peak) / p.seasonal_period
    return p.R0 * (1.0 + delta * np.cos(phase))


def build_W(M: sp.spmatrix, eps: float) -> sp.csr_matrix:
    """W = (1-eps) I + eps M (eq. W). M must be square, sparse and
    row-stochastic; its construction is left to the caller."""
    M = sp.csr_matrix(M)
    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError("M must be square")
    return sp.csr_matrix((1 - eps) * sp.identity(n, format="csr") + eps * M)


def _mix(W: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """W @ x for x of shape (R, n); rows of x are independent runs."""
    return (W @ x.T).T


def _penalty_cost(N_arr, u, dt: float, kind: str):
    """Population-weighted suppression-cost density (eq. costZ) for
    suppression exponent u (so the multiplier is exp(-u); u = alpha*Z for
    the rule/oracle-driven scenarios, u = -log(const_mult) for a constant
    multiplier -- same u either way, which is what makes a constant run
    directly comparable to a rule-driven run at matched effort), under one
    of three functional forms. All three agree to first order (exp(u)-1 =
    u + u^2/2 + ... near u=0), so they diverge only where suppression
    effort is large -- exactly the regime the paper's cost-convexity
    claims are about."""
    if kind == "exp":
        return N_arr * (np.exp(u) - 1.0) * dt
    if kind == "linear":
        return N_arr * u * dt
    if kind == "quadratic":
        return N_arr * 0.5 * u ** 2 * dt
    raise ValueError(f"unknown penalty kind: {kind!r}")


class _Ring:
    """Fixed-length circular history buffer for one (R, n) array per step."""

    def __init__(self, length: int, shape: tuple):
        self.length = max(length, 1)
        self.buf = np.zeros((self.length,) + shape)

    def push(self, t: int, value: np.ndarray) -> None:
        self.buf[t % self.length] = value

    def get(self, t: int, delay_steps: int) -> np.ndarray:
        return self.buf[(t - delay_steps) % self.length]


# ----------------------------------------------------------------------
def simulate(params: Params, M: sp.spmatrix, eps: float, N, n_steps: int,
             rng: np.random.Generator, rng_obs: "np.random.Generator | None" = None,
             n_reps: int = 1, seed_nodes=None, seed_X_SI=10.0,
             x_AI0_frac: "float | np.ndarray" = 0.0,
             x_invade: float = 0.01, track_trajectory: bool = False,
             oracle: bool = False, const_mult: "float | None" = None,
             track_iota: bool = False, burn_in_steps: int = 0,
             policy_fn=None, track_policy: bool = False):
    """
    Stochastic tau-leaping simulation of eqs. XSNdim/XSIdim/XAIdim
    (transmission), report/iotahat (reporting and the direct incidence
    estimator), xSNhat/xSIhat/xAIhat (prevalence nowcast) and pool (spatial
    pooling), with hat_iota_i(t) := hat_phi_i(t) as in Section 2.4.

    oracle=True drives the behavioural reduction factor
    e^{-alpha*Z_i(t-delay)} with the true, contemporaneous exposure
    phi_i (eq. phi) instead of the reported threat hat_iota_i -- the
    perfect-information counterfactual of Section 5.5. The reporting/nowcast
    pipeline still runs in this case (same code path either way, for a
    single well-tested loop rather than two); its output (hat_iota, the
    nowcast state) is simply not read by the transmission rate or the cost
    when oracle=True.

    const_mult, if given, overrides *both* of the above: the reduction
    factor is a fixed constant (const_mult) instead of e^{-alpha*Z_i(t)},
    independent of reporting, behaviour or alpha entirely -- a scripted,
    always-on suppression level, e.g. for comparing against what the
    reporting-driven system achieves. Takes priority over oracle if both
    are set. Cost is still priced against no suppression the same way
    (1/multiplier - 1), just with the constant multiplier in place of
    exp(-alpha*Z): N_i*(1/const_mult - 1)*dt, a constant rate rather than
    the time-varying cost of the reporting-driven scenarios.

    policy_fn, if given (and const_mult is None), replaces the fixed
    e^{-alpha*Z_i(t)} rule with a learned or scripted multiplier:
    mult = policy_fn(Z). oracle and policy_fn are independent axes (fork
    change from ../code/model.py, where they were mutually exclusive):
    oracle picks which signal Z is -- the true, contemporaneous phi_i (eq.
    phi) or the reported hat_iota_i -- and policy_fn, if given, picks how Z
    maps to a multiplier, in place of the closed-form e^{-alpha*Z}. Called
    on the same delayed Z the rule itself would use (for the transmission
    rate) and again on the current, undelayed Z (for the cost, matching how
    the rule's own mult_now is computed) -- Z = phi_i or hat_iota_i in both
    places according to oracle, exactly as for the rule.
    policy_fn must be a pure, memoryless function of its input array (same
    shape as Z, values in (0,1)) -- evaluating it on a delayed observation
    is only equivalent to having decided and cached an action back then if
    it carries no hidden state between calls, exactly as e^{-alpha*Z} has
    none. track_policy=True additionally records, every step and regardless
    of branch (const_mult/policy_fn/rule-or-oracle alike): mult_now (the
    current, undelayed reduction factor -- the same quantity cum_suppression
    accumulates, eq. costZ's Z_i(t)) and cost_now/new_SN/new_AI; when
    policy_fn is given it also records the two observations passed to it
    (obs_delayed_SN, obs_delayed_AI, obs_current). This lets a caller
    reconstruct a reward signal for RL training (the original purpose) or
    just inspect the reduction factor's own distribution over a run,
    without model.py needing to know about either use case's specifics.

    Either way, `cum_cost` accumulates the population-weighted suppression
    cost priced against no suppression at all (eq. costZ),
    N_i*(exp(alpha*Z_i(t)) - 1)*dt by default (params.penalty="exp"; "linear"
    and "quadratic" are also available, see Params.penalty), using the
    *current*, undelayed Z_i(t)
    (the cost borne by residents right now, from whatever is currently
    driving their behaviour) -- not the delayed value used in the
    transmission rate itself. `cum_infections` (already tracked) is the
    disease-burden accumulator B^Z of eq. burden, for whichever Z this run
    used. `cum_suppression` accumulates N_i*(1 - multiplier)*dt using that
    same current multiplier (1 - const_mult, or 1 - exp(-alpha*Z_i(t))) --
    dividing its sum over nodes by (total population * *measured*, i.e.
    post-burn-in, time) gives the population- and time-averaged suppression
    level, directly comparable across scenarios (e.g. against a constant
    suppression run).

    burn_in_steps excludes the first that many steps from cum_infections,
    cum_symptomatic, cum_cost and cum_suppression (though not from
    track_trajectory, which still records the full run including burn-in):
    the reporting/nowcast pipeline starts at zero and the epidemic starts at
    its seeded initial state, so the earliest steps reflect both settling
    rather than steady operation, and belong in the plotted trajectory but
    not in a summary statistic meant to characterise ongoing behaviour.

    N : scalar or (n,)-shaped array of node populations. x_AI0_frac sets
    every node's X_AI(0) to that fraction of N_i, a pre-existing
    asymptomatic reservoir present everywhere from the start rather than a
    disease introduced at a single node; may be a scalar (the same fraction
    at every node) or an (n,)-shaped array (a fixed, node-specific fraction
    -- e.g. sweep.py draws one once from a heterogeneous distribution and
    reuses it across every call, so spatial coupling has genuine cross-node
    variation to act on). seed_nodes/seed_X_SI instead place an initial
    symptomatic case count at specific nodes only, for studying spread from
    a point of introduction rather than an already-endemic starting state.
    The two are independent and may be used together or separately.
    Returns final compartment counts, a per-run invasion flag, cum_infections
    (= B^Z, new SN + new AI), cum_symptomatic (new SN only -- the
    symptomatic-disease-episode subset of B^Z, the social burden borne by
    residents rather than the full transmission reservoir), cum_cost
    (= Cost^Z), cum_suppression, and cum_active_time (fork addition:
    population-weighted node-time, N_i*dt summed wherever mult_now<1 --
    "how long was any suppression on", as opposed to cum_suppression's
    "how much did it reduce transmission by, weighted by how long"; the two
    coincide only for a policy whose multiplier takes exactly one non-1.0
    value, e.g. a bang-bang trigger, not for the continuous rule); pass
    track_trajectory=True for coarse
    mean-prevalence time series, traj ((X_SI+X_AI)/N, total infectious) and
    traj_symptomatic ((X_SN+X_SI)/N, currently symptomatic).

    track_iota=True (diagnostic only, no effect on the simulation itself)
    additionally records, every step, per node: iota_hat_traj (the current
    value of the iota_hat variable, i.e. hat_iota_i(t) := hat_phi_i(t) as
    actually computed, eq. pool -- the *estimated* force of infection),
    exponent_traj (-alpha * iota_hat_hist.get(t, d_SN), the delayed value
    actually plugged into exp() for rate_SN), and phi_true_traj
    (phi_i(t) = sum_j W_ij x_j(t), eq. phi -- the *true*, undelayed force
    of infection, built from the actual compartment counts rather than
    reports). iota_hat_traj vs. phi_true_traj is the estimated-vs-actual
    comparison; exponent_traj is what that estimate turns into once
    delayed and scaled by alpha inside the transmission rate. Use n_reps=1
    for this: averaging over replicates would smooth away exactly the
    step-to-step behaviour this is meant to expose.
    """
    p = params
    W = build_W(M, eps)
    n = W.shape[0]
    R = n_reps
    N_arr = np.broadcast_to(np.atleast_1d(np.asarray(N, dtype=float)), (n,)).copy()

    if p.rho > 0:
        # per-node reporting-noise threshold (eq. threshold): sum_j W_ij^2/N_j
        # is row i's squared mixing weights, each scaled by that neighbour's
        # own noise variance (eq. iotasd, 1/N_j) -- a node whose weight is
        # spread across many neighbours has a smaller sum here, and so a
        # smaller threshold, than one concentrated on few
        row_noise = np.asarray(W.multiply(W) @ (1.0 / N_arr)).ravel()
        c_thresh = p.kappa * np.sqrt(p.nu0 / (p.rho ** 2 * p.Delta) * row_noise)
    else:
        c_thresh = np.zeros(n)   # rho=0: hat_iota_{i,k} is 0 by convention
                                  # (eq. iotahat), so no signal ever reaches
                                  # the nowcast and the threshold is moot

    if rng_obs is None:
        rng_obs = np.random.default_rng(int(rng.integers(1 << 31)))

    d_SN = max(int(round(p.Tdel / p.dt)), 0)
    d_AI = max(int(round((p.Tdel + 1.0 / p.theta) / p.dt)), 0)
    d_th = max(int(round((1.0 / p.theta) / p.dt)), 0)
    hist_len = max(d_SN, d_AI, d_th) + 1
    win_steps = max(int(round(p.Delta / p.dt)), 1)
    ratio = (1 - p.sigma) / p.sigma if p.sigma > 0 else 0.0

    X_SN = np.zeros((R, n))
    X_SI = np.zeros((R, n))
    X_AI = np.broadcast_to(x_AI0_frac * N_arr, (R, n)).copy()   # pre-existing
                                    # asymptomatic reservoir, present at every
                                    # node from t=0 rather than introduced
                                    # through a single point of entry
    if seed_nodes is not None:
        X_SI[:, seed_nodes] = seed_X_SI

    Xh_SN = np.zeros((R, n))   # nowcast state, hat x^SN
    Xh_SI = np.zeros((R, n))   # nowcast state, hat x^SI
    Xh_AI = np.zeros((R, n))   # nowcast state, hat x^AI

    iota_hat = np.zeros((R, n))       # hat_iota_i(t) := hat_phi_i(t) (eq. pool),
                                       # the pooled reported threat -- recomputed
                                       # every step, drives transmission and cost
    iota_win_hat = np.zeros((R, n))   # hat_iota_{i,k} (eq. iotahat), the per-window
                                       # direct incidence estimate -- held constant
                                       # within a window, drives the nowcast ODE only
                                       # (eqs. xSNhat/xAIhat)
    win_new_SN = np.zeros((R, n))     # incidence accumulator for the current window

    x_I_hist = _Ring(hist_len, (R, n))       # (X_SI + X_AI) / N
    S_frac_hist = _Ring(hist_len, (R, n))    # 1 - x_SN - x_I
    iota_hat_hist = _Ring(hist_len, (R, n))
    iota_win_hat_hist = _Ring(hist_len, (R, n))
    xhat_I_hist = _Ring(hist_len, (R, n))    # Xhat_SI + Xhat_AI (already per-capita)

    ever = np.zeros((R, n), dtype=bool)
    cum_infections = np.zeros((R, n))   # running total of new_SN + new_AI (= B^Z)
    cum_symptomatic = np.zeros((R, n))  # running total of new_SN only -- new
                                         # symptomatic disease episodes, the
                                         # social-burden-facing subset of B^Z
    cum_cost = np.zeros((R, n))         # running suppression cost (= Cost^Z, eq. costZ)
    cum_suppression = np.zeros((R, n))  # running N_i*(1-multiplier)*dt, for the
                                         # population/time-averaged suppression level
    cum_active_time = np.zeros((R, n))  # running N_i*dt wherever mult_now<1 --
                                         # population-weighted node-time under ANY
                                         # suppression, distinct from cum_suppression's
                                         # effect-weighted effort: meaningful for a
                                         # policy (e.g. a threshold trigger) where "how
                                         # long was suppression on" matters in its own
                                         # right, not just how much it reduced transmission
    cum_episodes = np.zeros((R, n), dtype=np.int64)  # count of mult_now<1 "on"
                                         # episodes per node (transitions from off to
                                         # on), for alert-fatigue-relevant statistics
                                         # distinct from total time in force: mean
                                         # episode length = (cum_active_time/N_i) /
                                         # cum_episodes
    prev_active = np.zeros((R, n), dtype=bool)  # unsuppressed at t=0 by construction
    traj = [] if track_trajectory else None
    traj_symptomatic = [] if track_trajectory else None
    iota_hat_traj = [] if track_iota else None
    exponent_traj = [] if track_iota else None
    phi_true_traj = [] if track_iota else None
    xhat_I_traj = [] if track_iota else None
    x_I_traj = [] if track_iota else None
    obs_SN_traj = [] if track_policy else None
    obs_AI_traj = [] if track_policy else None
    obs_now_traj = [] if track_policy else None
    mult_now_traj = [] if track_policy else None
    cost_now_traj = [] if track_policy else None
    new_infections_traj = [] if track_policy else None

    for t in range(n_steps):
        x_SN = X_SN / N_arr
        x_I = (X_SI + X_AI) / N_arr
        S_frac = 1.0 - x_SN - x_I
        x_I_hist.push(t, x_I)
        S_frac_hist.push(t, S_frac)
        iota_hat_hist.push(t, iota_hat)
        iota_win_hat_hist.push(t, iota_win_hat)
        xhat_I_hist.push(t, Xh_SI + Xh_AI)

        # ---- delayed force-of-infection terms (eqs. XSNdim, XAIdim) ----
        phi_SN = _mix(W, x_I_hist.get(t, d_SN))
        phi_AI = _mix(W, x_I_hist.get(t, d_AI))
        S_SN = S_frac_hist.get(t, d_SN) * N_arr
        S_AI = S_frac_hist.get(t, d_AI) * N_arr

        if track_iota:
            iota_hat_traj.append(iota_hat.copy())
            exponent_traj.append(-p.alpha * iota_hat_hist.get(t, d_SN))
            phi_true_traj.append(_mix(W, x_I))   # phi_i(t) (eq. phi), true and undelayed
            xhat_I_traj.append((Xh_SI + Xh_AI).copy())   # nowcast state, pre-pooling, pre-delay
            x_I_traj.append(x_I.copy())                  # true state, pre-pooling, pre-delay
        if const_mult is not None:
            mult_SN = mult_AI = const_mult
        else:
            # oracle selects which signal drives this scenario (true phi_i,
            # the perfect-information counterfactual of Section 5.5, or the
            # reported hat_iota_i); policy_fn, if given, selects how that
            # signal maps to a multiplier (a learned/scripted policy in
            # place of the closed-form e^{-alpha*Z} rule) -- independent
            # axes, so a policy_fn can now be evaluated against either
            # signal, not just the reported one.
            obs_SN = phi_SN if oracle else iota_hat_hist.get(t, d_SN)
            obs_AI = phi_AI if oracle else iota_hat_hist.get(t, d_AI)
            if policy_fn is not None:
                mult_SN = policy_fn(obs_SN)
                mult_AI = policy_fn(obs_AI)
                if track_policy:
                    obs_SN_traj.append(obs_SN.copy())
                    obs_AI_traj.append(obs_AI.copy())
            else:
                mult_SN = np.exp(-p.alpha * obs_SN)
                mult_AI = np.exp(-p.alpha * obs_AI)

        # R0 evaluated at the same delayed argument as phi_SN/phi_AI: both
        # describe conditions at the moment of the contact that produces the
        # case appearing at t, not conditions now (eq. R0seasonal).
        R0_SN = _R0_of_t(p, (t - d_SN) * p.dt)
        R0_AI = _R0_of_t(p, (t - d_AI) * p.dt)
        rate_SN = p.sigma * R0_SN * mult_SN * phi_SN
        rate_AI = (1 - p.sigma) * R0_AI * mult_AI * phi_AI

        new_SN = rng.poisson(np.clip(rate_SN, 0, None) * np.clip(S_SN, 0, None) * p.dt)
        new_AI = rng.poisson(np.clip(rate_AI, 0, None) * np.clip(S_AI, 0, None) * p.dt)

        # cap by *current* susceptibles so S cannot go negative even though
        # the rate above uses the delayed susceptible fraction, per eq. XAIdim
        S_now = np.maximum(N_arr - X_SN - X_SI - X_AI, 0.0)
        new_SN = np.minimum(new_SN, S_now)
        new_AI = np.minimum(new_AI, np.maximum(S_now - new_SN, 0.0))

        mature = rng.binomial(X_SN.astype(np.int64), 1 - np.exp(-p.theta * p.dt))
        recover = rng.binomial(X_SI.astype(np.int64), 1 - np.exp(-1.0 * p.dt))
        clear = rng.binomial(X_AI.astype(np.int64), 1 - np.exp(-p.gamma * p.dt))

        X_SN = X_SN + new_SN - mature
        X_SI = X_SI + mature - recover
        X_AI = X_AI + new_AI - clear
        np.clip(X_SN, 0, N_arr, out=X_SN)
        np.clip(X_SI, 0, N_arr, out=X_SI)
        np.clip(X_AI, 0, N_arr, out=X_AI)

        win_new_SN += new_SN
        ever |= ((X_SI + X_AI) / N_arr) >= x_invade

        # ---- suppression cost (eq. costZ), priced against no suppression:
        # current, undelayed Z_i(t) -- what's actually driving behaviour
        # right now -- not the delayed value the transmission rate used above.
        # const_mult is time-invariant, so its cost is just a constant rate;
        # its implied exponent u=-log(const_mult) is priced under the same
        # p.penalty as the rule/oracle branch, so a constant run is directly
        # comparable to a rule-driven run at matched effort regardless of
        # penalty form (both reduce to N*(1/mult-1)*dt under "exp").
        if const_mult is not None:
            mult_now = const_mult
            cost_now = _penalty_cost(N_arr, -np.log(const_mult), p.dt, p.penalty)
        else:
            cost_signal = _mix(W, x_I) if oracle else iota_hat
            if policy_fn is not None:
                mult_now = policy_fn(cost_signal)
                cost_now = _penalty_cost(N_arr, -np.log(mult_now), p.dt, p.penalty)
                if track_policy:
                    obs_now_traj.append(cost_signal.copy())
            else:
                mult_now = np.exp(-p.alpha * cost_signal)
                cost_now = _penalty_cost(N_arr, p.alpha * cost_signal, p.dt, p.penalty)

        active_now = np.asarray(mult_now) < 1.0
        if t >= burn_in_steps:
            # excluded during burn-in (docstring above): the nowcast and the
            # epidemic are both still settling from their initial state, not
            # yet reflecting steady operation
            cum_infections += new_SN + new_AI
            cum_symptomatic += new_SN
            cum_cost += cost_now
            cum_suppression += N_arr * (1.0 - mult_now) * p.dt
            cum_active_time += N_arr * active_now * p.dt
            cum_episodes += (active_now & ~prev_active)
        prev_active = active_now

        if track_policy:
            cost_now_traj.append(cost_now.copy())
            new_infections_traj.append((new_SN + new_AI).copy())
            mult_now_traj.append(np.asarray(mult_now).copy())

        # ---- prevalence nowcast (eqs. xSNhat-xAIhat), Euler step ----
        iota_win_lag = iota_win_hat_hist.get(t, d_th)   # hat_iota_j^win(t - 1/theta)
        Xh_SN = Xh_SN + p.dt * (iota_win_hat - p.theta * Xh_SN)
        Xh_SI = Xh_SI + p.dt * (p.theta * Xh_SN - Xh_SI)
        Xh_AI = Xh_AI + p.dt * (ratio * iota_win_lag - p.gamma * Xh_AI)

        # ---- spatial pooling (eq. pool, eq. nowcastclip) ----
        # non-negativity and reporting-noise suppression (c_thresh, eq.
        # threshold) are both enforced once, here, on the pooled result --
        # not on the per-node state above
        iota_hat = np.clip(_mix(W, xhat_I_hist.get(t, d_SN)) - c_thresh, 0, None)

        # ---- reporting, every `window` steps (eqs. report, iotahat) ----
        if (t + 1) % win_steps == 0:
            C = rng_obs.poisson(p.rho * win_new_SN + p.nu0 * N_arr * p.Delta)
            if p.c_min > 0:
                C = np.where(C < p.c_min, 0, C)
            if p.rho > 0:
                # unbiased for the true window incidence rate, not clipped at
                # zero: the nowcast state (Xh_SN/Xh_SI/Xh_AI) is what needs to
                # stay non-negative, and it already is (clipped below), every
                # step. Clipping this per-window flow estimate itself, before
                # it drives the nowcast, would introduce a one-sided bias
                # every window instead of letting positive and negative noise
                # partially cancel first.
                iota_win = (C - p.nu0 * N_arr * p.Delta) / (p.rho * N_arr * p.Delta)
            else:
                iota_win = np.zeros_like(C, dtype=float)
            iota_win_hat = iota_win    # held constant as the nowcast input until
                                        # the next window boundary
            win_new_SN[:] = 0.0

        if track_trajectory and t % 5 == 0:
            traj.append(((X_SI + X_AI) / N_arr).mean(axis=1))
            traj_symptomatic.append(((X_SN + X_SI) / N_arr).mean(axis=1))

    out = dict(X_SN=X_SN, X_SI=X_SI, X_AI=X_AI, ever=ever, iota_hat=iota_hat,
               cum_infections=cum_infections, cum_symptomatic=cum_symptomatic,
               cum_cost=cum_cost, cum_suppression=cum_suppression,
               cum_active_time=cum_active_time, cum_episodes=cum_episodes)
    if track_trajectory:
        out["traj"] = np.array(traj)
        out["traj_symptomatic"] = np.array(traj_symptomatic)
    if track_iota:
        out["iota_hat_traj"] = np.array(iota_hat_traj)      # (n_steps, R, n)
        out["exponent_traj"] = np.array(exponent_traj)      # (n_steps, R, n)
        out["phi_true_traj"] = np.array(phi_true_traj)      # (n_steps, R, n)
        out["xhat_I_traj"] = np.array(xhat_I_traj)          # (n_steps, R, n)
        out["x_I_traj"] = np.array(x_I_traj)                # (n_steps, R, n)
    if track_policy:
        out["obs_SN_traj"] = np.array(obs_SN_traj)                  # (n_steps, R, n)
        out["obs_AI_traj"] = np.array(obs_AI_traj)                  # (n_steps, R, n)
        out["obs_now_traj"] = np.array(obs_now_traj)                # (n_steps, R, n)
        out["mult_now_traj"] = np.array(mult_now_traj)              # (n_steps, R, n)
        out["cost_now_traj"] = np.array(cost_now_traj)              # (n_steps, R, n)
        out["new_infections_traj"] = np.array(new_infections_traj)  # (n_steps, R, n)
    return out


# ----------------------------------------------------------------------
# Placeholder only, for smoke-testing this module. M's real construction
# (distance kernel, gravity/radiation model, ward adjacency, ...) is not
# yet decided -- do not use this for actual results.
def _placeholder_ring_M(n: int) -> sp.csr_matrix:
    idx = np.arange(n)
    rows = np.repeat(idx, 2)
    cols = np.concatenate([(idx - 1) % n, (idx + 1) % n])
    data = np.full(2 * n, 0.5)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


if __name__ == "__main__":
    n = 20
    M = _placeholder_ring_M(n)
    params = Params()
    result = simulate(params, M, eps=0.2, N=5000, n_steps=1500,
                       rng=np.random.default_rng(0), n_reps=4,
                       seed_nodes=[0], seed_X_SI=50.0, track_trajectory=True)
    prevalence = (result["X_SI"] + result["X_AI"]) / 5000
    print("final mean prevalence:", prevalence.mean())
    print("fraction of runs with >=1 node ever invaded:",
          result["ever"].any(axis=1).mean())
    print("trajectory shape:", result["traj"].shape)
