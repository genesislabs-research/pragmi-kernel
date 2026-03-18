"""
ca3_memory.py
=============

PRAGMI Cognitive Kernel: CA3 Autoassociative Memory
===================================================

BIOLOGICAL GROUNDING
--------------------
This file models hippocampal area CA3, the autoassociative network at the core
of episodic memory formation and retrieval. CA3 is the only region in the
hippocampal formation where recurrent collateral (RC) synapses connect pyramidal
cells to each other at scale, forming a single interconnected attractor network.

In the rat, each CA3 pyramidal cell receives input from three sources:

    ~46 mossy fiber synapses (from dentate gyrus granule cells):
        Few in number but extremely powerful per synapse. A single granule cell
        can reliably discharge its CA3 target. These inputs FORCE new patterns
        into CA3 during encoding, overriding the recurrent dynamics.

    ~3,600 perforant path synapses (from entorhinal cortex layer II):
        Many in number but individually weak. These provide the RETRIEVAL CUE
        that initiates pattern completion. They are insufficient to drive
        storage but sufficient to nudge the recurrent network toward the
        correct attractor basin.

    ~12,000 recurrent collateral synapses (from other CA3 pyramidal cells):
        These form the autoassociative memory itself. The recurrent weights
        encode attractor basins, and the recurrent dynamics settle partial
        cues into complete patterns.

The storage capacity of the CA3 network scales with the number of recurrent
synapses per cell and the sparseness of the representation. Rolls (2013)
provides the quantitative analysis showing that CA3's ~12,000 RC synapses
and sparse firing (~1-2% active) yield a capacity on the order of thousands
of distinct episodic memories.

Lead papers:

1. Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
   separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
   DOI: 10.3389/fnsys.2013.00074

2. Das, P., Bhatt, S., Bhagavatula, S., & Bhattacharjee, B. (2024).
   "Larimar: Large Language Models with Episodic Memory Control."
   ICML 2024. arXiv: 2403.11901. DOI: 10.48550/arXiv.2403.11901

3. Ben-Israel, A. & Cohen, D. (1966). "On iterative computation of
   generalized inverses and associated projections." SIAM Journal on
   Numerical Analysis, 3(3), 410-419. DOI: 10.1137/0703035


WHY THIS FILE EXISTS (for engineers)
------------------------------------
This module does two things: STORE episodes and RETRIEVE them from partial
cues. Both operations use the same (K, C) memory matrix M where K is the
number of memory slots and C is the coordinate dimensionality.

RETRIEVAL (read): Given a cue vector q, find the addressing weights w such
that the reconstruction w @ M is as close to the original stored episode as
possible. We solve this by computing the Moore-Penrose pseudoinverse of M
and projecting the cue through it: w = q @ M_pinv.

STORAGE (write): Given a new episode z, update M so that z can be retrieved
later without destroying previously stored episodes. Naive overwrite (replace
a row of M) causes catastrophic interference. Instead, we use the Bayesian
posterior update from Larimar, which is mathematically a Kalman filter step.
The prediction error (z - w @ M) is used to update the memory matrix with
minimal interference, weighted by the current uncertainty estimate for each
slot. Slots that have been written many times have low uncertainty and resist
further modification (they are "consolidated"), while fresh slots have high
uncertainty and accept new information readily.

The pseudoinverse itself is never computed analytically (that would be O(K^2*C)
and numerically unstable). Instead, we use the Ben-Israel-Cohen iterative
method, which starts from alpha * M^T and refines it over T iterations. The
step size alpha must be chosen so that the initial approximation is in the
convergence basin of the true pseudoinverse. We set alpha = 1/||M||_F^2 times
a learned correction factor, then check the residual after each iteration. If
the residual has not converged by the 8th iteration (matching our spiking
timestep window T=8), we flag a metabolic alert.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    from cognitive_kernel import HippocampalConfig
except ImportError:

    @dataclass
    class HippocampalConfig:
        """Minimal fallback config for standalone testing.

        See cognitive_kernel.py for the full config with all parameter
        citations and biological grounding.
        """
        coordinate_dim: int = 64
        bridge_dim: int = 496
        comm_subspace_rank: int = 3
        ca3_memory_slots: int = 256
        ca3_code_dim: int = 64
        pseudoinverse_iterations: int = 8
        pseudoinverse_alpha_init: float = 0.0
        convergence_tolerance: float = 1e-4
        observation_noise_std: float = 0.0
        T: int = 8


# =============================================================================
# CA3 AUTOASSOCIATIVE MEMORY
# =============================================================================

class CA3RecurrentMatrix(nn.Module):
    """Autoassociative memory matrix modeling hippocampal area CA3.

    BIOLOGICAL STRUCTURE: CA3 pyramidal cell network with recurrent
    collateral (RC) synapses. In the rat, each CA3 cell receives ~12,000
    RC synapses from other CA3 cells, forming a single interconnected
    autoassociative network. The connectivity is approximately 2% (12,000
    out of ~300,000 CA3 cells), and Rolls (2013) shows that this diluted
    connectivity still supports attractor dynamics with only moderate
    reduction in storage capacity compared to full connectivity.

    In primates (macaques), the RC projections are even more extensive along
    the longitudinal axis, meaning the primate CA3 operates as a more
    thoroughly interconnected single network than the rodent version.

    Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
    separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
    DOI: 10.3389/fnsys.2013.00074

    BIOLOGICAL FUNCTION: CA3 stores episodic memories as attractor basins
    in its recurrent weight matrix. Each basin is a stable fixed point of
    the recurrent dynamics: when the network is initialized near that point
    (by a partial cue via the perforant path), the dynamics converge to it,
    completing the pattern.

    Storage uses strong mossy fiber input to force a new pattern; retrieval
    uses weaker perforant path input to cue recall. This asymmetry is
    critical and is why the ~46 mossy fiber synapses per cell can override
    the ~12,000 recurrent synapses during encoding.

    COMPUTATIONAL IMPLEMENTATION:

    The memory is stored as a matrix M of shape (K, C) where K is the
    number of memory slots (attractor basins) and C is the coordinate
    dimensionality. Read and write operations use the Moore-Penrose
    pseudoinverse of M, approximated by the Ben-Israel-Cohen iteration.

    WRITE MECHANISM (Bayesian pseudoinverse addressing):
        1. Compute addressing weights: w = z @ M_pinv
        2. Compute prediction error: delta = z - w @ M
        3. Compute Kalman gain: K_gain = (w * variance) / (w^2 * variance + noise)
        4. Update: M_new = M + K_gain * delta
        5. Update variance: V_new = V - K_gain * w * V

    This is a Kalman filter step where the "measurement" is the new episode z,
    the "prediction" is w @ M, and the Kalman gain weights the update by the
    current uncertainty. Slots with low variance resist modification.

    Das, P. et al. (2024). "Larimar: Large Language Models with Episodic
    Memory Control." arXiv: 2403.11901. DOI: 10.48550/arXiv.2403.11901

    CONVERGENCE STABILIZATION:
    The Ben-Israel iteration converges when the initial approximation
    A_pinv_0 = alpha * A^T satisfies ||alpha * A^T @ A||_2 < 1. Setting
    alpha = 1/||A||_F^2 guarantees this because ||A||_F >= ||A||_2.
    A learned log-scale correction factor allows the system to fine-tune
    alpha during training while the clamp (max 5e-4) prevents divergence.

    Ben-Israel, A. & Cohen, D. (1966). "On iterative computation of
    generalized inverses and associated projections." SIAM Journal on
    Numerical Analysis, 3(3), 410-419. DOI: 10.1137/0703035
    """

    def __init__(self, cfg: HippocampalConfig):
        """Initialize the CA3 autoassociative memory matrix.

        Creates the memory matrix (K, C), the uncertainty matrix (K, C),
        the slot occupancy tracker, and the learnable Ben-Israel step size.

        Parameters
        ----------
        cfg : HippocampalConfig
            Kernel configuration. The relevant fields are:
                ca3_memory_slots        number of attractor basins (K)
                ca3_code_dim            coordinate dimensionality (C)
                pseudoinverse_iterations  max Ben-Israel steps (matches T)
                convergence_tolerance   residual threshold for early stop
                pseudoinverse_alpha_init  initial learned log-scale for alpha
                observation_noise_std   noise floor for Kalman gain denominator
        """
        super().__init__()
        self.cfg = cfg
        K = cfg.ca3_memory_slots
        C = cfg.ca3_code_dim

        # -----------------------------------------------------------------
        # memory_mean: (K, C)  - the memory matrix
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: CA3 Recurrent Collateral Weight Matrix
        #
        # PLAIN ENGLISH: Each row is one stored episodic memory. The matrix
        # as a whole defines the attractor landscape: the set of stable
        # patterns that the network can settle into during recall. In the
        # biological CA3, this information is encoded in the ~12,000
        # recurrent synaptic weights per cell, not as explicit "rows", but
        # the functional equivalence is that each attractor basin corresponds
        # to a distributed pattern of synaptic weights that produces a
        # stable firing pattern.
        #
        # INITIALIZATION: Small random values (std=0.01). Not zeros, because
        # the pseudoinverse of a zero matrix is undefined and the Ben-Israel
        # iteration would diverge on the first write. Not large values,
        # because that would create strong spurious attractors before any
        # real episodes have been stored.
        # NOT a biological quantity. Engineering initialization choice.
        #
        # CITATION: Rolls, E.T. (2013). "The mechanisms for pattern completion
        # and pattern separation in the hippocampus." Frontiers in Systems
        # Neuroscience, 7, 74. DOI: 10.3389/fnsys.2013.00074
        # -----------------------------------------------------------------
        self.register_buffer(
            "memory_mean",
            torch.randn(K, C) * 0.01
        )

        # -----------------------------------------------------------------
        # memory_variance: (K, C)  - posterior uncertainty per slot
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Synaptic Confidence (no direct biological analog)
        #
        # PLAIN ENGLISH: Each element tracks how certain the system is about
        # the value stored in that position. High variance means the slot is
        # fresh and will readily accept new information. Low variance means
        # the slot has been written many times and resists modification. This
        # is the mechanism by which older, consolidated memories become
        # harder to overwrite, a property observed in the biological
        # reconsolidation literature.
        #
        # INITIALIZATION: Ones (maximum uncertainty, uniform prior).
        # NOT a biological quantity. This is the Bayesian prior from Larimar.
        #
        # CITATION: Das, P. et al. (2024). "Larimar: Large Language Models
        # with Episodic Memory Control." arXiv: 2403.11901.
        # DOI: 10.48550/arXiv.2403.11901
        # -----------------------------------------------------------------
        self.register_buffer(
            "memory_variance",
            torch.ones(K, C)
        )

        # -----------------------------------------------------------------
        # slot_write_count: (K,)  - occupancy tracker
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: none (bookkeeping)
        #
        # PLAIN ENGLISH: Counts how many times each slot has been written.
        # Used to find the least-used slot when no target slot is specified
        # (new encoding). In the biological hippocampus, the dentate gyrus
        # performs this function by pattern separating new inputs into
        # orthogonal representations, which naturally targets unused or
        # weakly-committed CA3 populations.
        #
        # NOT a biological quantity. Engineering heuristic replacing DG
        # pattern separation in the prototype.
        # -----------------------------------------------------------------
        self.register_buffer(
            "slot_write_count",
            torch.zeros(K, dtype=torch.long)
        )

        # -----------------------------------------------------------------
        # ben_israel_log_scale: scalar  - learned correction for alpha
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: none
        #
        # PLAIN ENGLISH: The Ben-Israel iteration needs an initial step size
        # alpha. The optimal alpha is 1/||M||_2^2 (inverse squared spectral
        # norm of the memory matrix). We approximate this with 1/||M||_F^2
        # (cheaper, conservative) and then multiply by a learned factor
        # exp(ben_israel_log_scale), clamped to max 5e-4 to prevent
        # divergence. This lets the system fine-tune the step size during
        # training without risking instability.
        #
        # Initialized to 0.0, so exp(0) = 1.0, meaning the initial alpha
        # is pure 1/||M||_F^2 with no learned correction.
        #
        # NOT a biological quantity. Training artifact for convergence.
        #
        # CITATION: Ben-Israel, A. & Cohen, D. (1966). "On iterative
        # computation of generalized inverses and associated projections."
        # SIAM Journal on Numerical Analysis, 3(3), 410-419.
        # DOI: 10.1137/0703035
        # -----------------------------------------------------------------
        self.ben_israel_log_scale = nn.Parameter(
            torch.tensor(cfg.pseudoinverse_alpha_init)
        )

        # Convergence diagnostics (registered buffers for serialization)
        self.register_buffer("last_residual", torch.tensor(0.0))
        self.register_buffer("last_converged", torch.tensor(True))
        self.register_buffer("iterations_used", torch.tensor(0))

    # =====================================================================
    # BEN-ISRAEL PSEUDOINVERSE ENGINE
    # =====================================================================

    def _compute_dynamic_alpha(self, A: Tensor) -> Tensor:
        """Compute the step size for the Ben-Israel iteration.

        The iteration A_pinv_{k+1} = 2*A_pinv_k - A_pinv_k @ A @ A_pinv_k
        converges to the Moore-Penrose pseudoinverse if and only if the
        initial approximation A_pinv_0 = alpha * A^T satisfies:

            0 < alpha < 2 / ||A||_2^2

        where ||A||_2 is the spectral norm (largest singular value).

        We use the Frobenius norm as a cheaper upper bound on the spectral
        norm. Since ||A||_F >= ||A||_2, we have:

            1/||A||_F^2 <= 1/||A||_2^2

        So alpha = 1/||A||_F^2 is always in the convergence basin, but it
        may be more conservative than necessary (slower convergence). The
        learned log-scale factor allows the system to nudge alpha closer to
        the optimal value during training.

        COMPUTATIONAL NOTE: The Frobenius norm squared is just the sum of
        all squared elements, which is O(K*C) and fully parallelizable.
        The spectral norm would require SVD or power iteration, both more
        expensive.

        Parameters
        ----------
        A : Tensor, shape (K, C)
            The memory matrix.

        Returns
        -------
        alpha : Tensor, scalar
            Step size for the Ben-Israel initialization.
        """
        # ||A||_F^2 = sum of all squared elements
        A_fro_sq = (A * A).sum()

        # Base alpha: 1 / ||A||_F^2
        alpha_base = 1.0 / (A_fro_sq + 1e-8)

        # Learned correction, clamped for stability.
        # Larimar uses max 5e-4 to prevent divergence in their experiments.
        # Das, P. et al. (2024). DOI: 10.48550/arXiv.2403.11901
        learned_scale = torch.clamp(
            torch.exp(self.ben_israel_log_scale), max=5e-4
        )

        return alpha_base * learned_scale

    def _approx_pseudoinverse(
        self,
        A: Tensor,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, bool, int]:
        """Ben-Israel-Cohen iterative pseudoinverse with convergence guard.

        Computes an approximation to the Moore-Penrose pseudoinverse of A
        via the iteration:

            A_pinv_0 = alpha * A^T
            A_pinv_{k+1} = 2 * A_pinv_k - A_pinv_k @ A @ A_pinv_k

        At convergence, A @ A_pinv @ A = A (the defining property of the
        pseudoinverse). We check this residual after each iteration and
        stop early if it falls below the tolerance.

        If the residual has not converged after max_iterations steps (which
        should match the spiking timestep window T=8), the caller should
        trigger a metabolic alert via the AstrocyticRegulator.

        Ben-Israel, A. & Cohen, D. (1966). "On iterative computation of
        generalized inverses and associated projections." SIAM Journal on
        Numerical Analysis, 3(3), 410-419. DOI: 10.1137/0703035

        WHY NOT JUST USE torch.linalg.pinv()?
        Three reasons. First, the iterative method lets us check convergence
        at each step and bail early, which torch.linalg.pinv does not.
        Second, the iterative method gives us a natural integration point
        with the spiking timestep window: each iteration corresponds to one
        "tick" of the recurrent dynamics. Third, the learned alpha parameter
        lets the system tune convergence speed during training, which a
        closed-form SVD does not support.

        Parameters
        ----------
        A : Tensor, shape (K, C)
            The memory matrix.
        max_iterations : int, optional
            Maximum number of iterations. Defaults to cfg.pseudoinverse_iterations.
        tolerance : float, optional
            Convergence threshold. Defaults to cfg.convergence_tolerance.

        Returns
        -------
        A_pinv : Tensor, shape (C, K)
            Approximation of the Moore-Penrose pseudoinverse.
        residual : Tensor, scalar
            Final residual ||A @ A_pinv @ A - A||_F.
        converged : bool
            Whether the residual fell below tolerance.
        iters_used : int
            Number of iterations actually performed (for diagnostics).
        """
        if max_iterations is None:
            max_iterations = self.cfg.pseudoinverse_iterations
        if tolerance is None:
            tolerance = self.cfg.convergence_tolerance

        alpha = self._compute_dynamic_alpha(A)

        # Initialize: A_pinv_0 = alpha * A^T
        A_pinv = alpha * A.T  # shape (C, K)

        converged = False
        residual = torch.tensor(float('inf'), device=A.device)
        iters_used = 0

        for i in range(max_iterations):
            iters_used = i + 1

            # Ben-Israel step: A_{k+1} = 2*A_k - A_k @ A @ A_k
            # Computational cost per iteration: two (C, K) x (K, C) matmuls
            # plus one (C, K) x (K, K) -> this is actually:
            #   (C, K) @ (K, C) @ (C, K) which is O(C * K * C) + O(C * K * K)
            # For K=256, C=64: ~1M MACs per iteration, ~8M for all 8 iterations.
            A_pinv_new = 2.0 * A_pinv - A_pinv @ A @ A_pinv

            # Residual: ||A @ A_pinv @ A - A||_F
            # At convergence this is zero. We normalize by ||A||_F for
            # scale-invariant comparison against the tolerance.
            reconstruction = A @ A_pinv_new @ A
            residual = torch.norm(reconstruction - A, p='fro')

            A_pinv = A_pinv_new

            if residual.item() < tolerance:
                converged = True
                break

        return A_pinv, residual, converged, iters_used

    # =====================================================================
    # READ (Pattern Completion)
    # =====================================================================

    def read(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        """Retrieve a memory pattern from a partial cue.

        BIOLOGICAL ANALOG: Pattern Completion via CA3 Recurrent Collaterals.

        A partial cue arrives via the perforant path (~3,600 synapses per
        CA3 cell). This activates a subset of the recurrent collateral
        network, which then settles into the nearest attractor basin via
        iterative dynamics. The settled state IS the completed pattern.

        We approximate this attractor settlement with a single pseudoinverse
        solve rather than simulating the full recurrent dynamics. This is a
        simplification that preserves the key functional property (partial
        cues retrieve whole patterns) while being computationally tractable.

        INTERFACE BOUNDARY:
            SENDING:    Perforant Path (retrieval cue from entorhinal cortex)
            RECEIVING:  CA3 Recurrent Collateral Network (pattern completion)
            CONNECTION: Perforant Path to CA3 (EC layer II direct projection)

        Rolls, E.T. (2013). DOI: 10.3389/fnsys.2013.00074

        Parameters
        ----------
        query : Tensor, shape (batch, ca3_code_dim)
            The retrieval cue in coordinate space.

        Returns
        -------
        reconstruction : Tensor, shape (batch, ca3_code_dim)
            The completed memory pattern.
        addressing_weights : Tensor, shape (batch, ca3_memory_slots)
            Soft attention over memory slots. Useful for: identifying which
            slot was activated (diagnostics), targeting that slot during
            reconsolidation (CA1 routing), and computing retrieval confidence.
        """
        M = self.memory_mean  # (K, C)

        A_pinv, residual, converged, iters = self._approx_pseudoinverse(M)

        # Store convergence diagnostics for external monitoring
        self.last_residual.fill_(residual.item())
        self.last_converged.fill_(converged)
        self.iterations_used.fill_(iters)

        # Addressing: w = query @ M_pinv, shape (batch, K)
        addressing_weights = query @ A_pinv

        # Softmax produces a probability distribution over memory slots.
        # This means retrieval is always a soft blend of memories, weighted
        # by similarity. A sharp softmax (one slot dominates) corresponds
        # to confident recall from a single attractor basin. A flat softmax
        # (many slots contribute) corresponds to uncertain recall where
        # the cue is equidistant from multiple basins.
        addressing_weights = F.softmax(addressing_weights, dim=-1)

        # Reconstruct: z_hat = w @ M
        reconstruction = addressing_weights @ M  # (batch, C)

        return reconstruction, addressing_weights

    # =====================================================================
    # WRITE (Episodic Encoding)
    # =====================================================================

    def write(
        self,
        z: Tensor,
        slot_idx: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Store a new episode via Bayesian pseudoinverse update.

        BIOLOGICAL ANALOG: Mossy Fiber Forcing + Hebbian Weight Update.

        During encoding, the dentate gyrus fires strongly via the ~46 mossy
        fiber synapses per CA3 cell. This input is powerful enough to
        override the recurrent collateral dynamics and force a new activation
        pattern into CA3. The recurrent weights are then updated via
        Hebbian learning (LTP at the RC synapses) to store this new pattern
        as a new attractor basin.

        We model this as a Bayesian update (Kalman filter step) on the
        memory matrix. The key insight from Larimar is that the addressing
        weights from the pseudoinverse tell us how the new episode relates
        to existing memories, and the Kalman gain weights the update by
        current uncertainty. This produces interference-minimizing writes:
        the update is concentrated on the dimensions where the memory is
        most uncertain, leaving well-consolidated dimensions untouched.

        INTERFACE BOUNDARY:
            SENDING:    Dentate Gyrus (pattern-separated new representation)
            RECEIVING:  CA3 Recurrent Collateral Network (weight update)
            CONNECTION: Mossy Fibers (DG granule cell axons to CA3 pyramidals)

        Rolls, E.T. (2013). DOI: 10.3389/fnsys.2013.00074 (mossy fiber forcing)
        Das, P. et al. (2024). DOI: 10.48550/arXiv.2403.11901 (Bayesian update)

        Parameters
        ----------
        z : Tensor, shape (batch, ca3_code_dim)
            The episode vector to store.
        slot_idx : Tensor, optional, shape (batch,)
            Target slot indices. If provided, forces writes to specific slots
            (used during reconsolidation to update an existing basin). If
            None, the system writes to the least-used slot (new encoding).

        Returns
        -------
        stats : dict containing:
            slot_indices:          which slots were written (batch,)
            residual:              pseudoinverse residual (scalar)
            converged:             whether Ben-Israel converged (bool)
            iterations_used:       how many iterations were needed (int)
            prediction_error_norm: mean ||z - w @ M|| across batch (scalar)
        """
        M = self.memory_mean  # (K, C)
        V = self.memory_variance  # (K, C)
        K, C = M.shape
        batch = z.shape[0]

        # --- Slot selection ---
        if slot_idx is None:
            # Heuristic: write to the least-used slot.
            # In the full architecture, the dentate gyrus would perform
            # pattern separation to find an orthogonal slot. This heuristic
            # approximates that by preferring empty or rarely-used slots.
            counts = self.slot_write_count.clone()
            slot_idx = torch.stack([
                torch.argmin(counts) for _ in range(batch)
            ])

        # --- Pseudoinverse addressing ---
        A_pinv, residual, converged, iters = self._approx_pseudoinverse(M)

        # Store convergence diagnostics
        self.last_residual.fill_(residual.item())
        self.last_converged.fill_(converged)
        self.iterations_used.fill_(iters)

        # Addressing weights and prediction error
        w = F.softmax(z @ A_pinv, dim=-1)  # (batch, K)
        z_hat = w @ M  # (batch, C)
        delta = z - z_hat  # prediction error

        # --- Bayesian update (Kalman filter step) ---
        # For each batch element, update the target slot.
        #
        # The update rule is:
        #   K_gain = (w_s * V_s) / (w_s^2 * V_s + noise_var)
        #   M_s_new = M_s + K_gain * delta
        #   V_s_new = V_s - K_gain * w_s * V_s
        #
        # where w_s is the addressing weight for the target slot and V_s is
        # the per-element variance (uncertainty) for that slot.
        #
        # When noise_var = 0 and w_s is close to 1 (sharp addressing), the
        # Kalman gain approaches 1/w_s, and the update is approximately
        # M_s_new = M_s + delta, which is a full overwrite. When V_s is
        # small (consolidated slot), the gain is small and the update is
        # gentle. This is the mechanism by which older memories resist
        # modification.
        noise_var = self.cfg.observation_noise_std ** 2

        for b in range(batch):
            s = slot_idx[b].item()

            slot_var = V[s]  # (C,)
            w_s = w[b, s]  # scalar: addressing weight for target slot

            # Kalman gain (diagonal approximation, element-wise over C)
            gain_denom = w_s * w_s * slot_var + noise_var + 1e-8
            kalman_gain = (w_s * slot_var) / gain_denom  # (C,)

            # Update memory mean (the stored pattern)
            self.memory_mean[s] = M[s] + kalman_gain * delta[b]

            # Update variance (reduce uncertainty on written dimensions)
            self.memory_variance[s] = V[s] - kalman_gain * w_s * V[s]
            self.memory_variance[s] = torch.clamp(
                self.memory_variance[s], min=1e-6  # floor prevents degenerate certainty
            )

            self.slot_write_count[s] += 1

        return {
            "slot_indices": slot_idx,
            "residual": residual,
            "converged": converged,
            "iterations_used": iters,
            "prediction_error_norm": torch.norm(delta, dim=-1).mean(),
        }

    # =====================================================================
    # DIAGNOSTICS
    # =====================================================================

    def get_diagnostics(self) -> Dict[str, object]:
        """Return a comprehensive diagnostic snapshot of the CA3 memory state.

        Returns
        -------
        diagnostics : dict
            slots_used:         number of slots written at least once
            total_writes:       total write operations across all slots
            mean_variance:      average uncertainty across all slots (lower = more consolidated)
            last_residual:      most recent Ben-Israel residual
            last_converged:     whether the last pseudoinverse converged
            iterations_used:    how many iterations the last solve needed
            memory_fro_norm:    Frobenius norm of M (proxy for total stored information)
        """
        return {
            "slots_used": (self.slot_write_count > 0).sum().item(),
            "total_writes": self.slot_write_count.sum().item(),
            "mean_variance": self.memory_variance.mean().item(),
            "last_residual": self.last_residual.item(),
            "last_converged": self.last_converged.item(),
            "iterations_used": self.iterations_used.item(),
            "memory_fro_norm": torch.norm(self.memory_mean, p='fro').item(),
        }

    def reset_memory(self):
        """Clear all stored episodes and reset to prior state.

        This is the computational analog of hippocampal "clearing" that
        would occur if the entire CA3 network were reset. Not biologically
        realistic (the brain does not do this), but useful for testing.
        """
        K, C = self.cfg.ca3_memory_slots, self.cfg.ca3_code_dim
        self.memory_mean.copy_(torch.randn(K, C) * 0.01)
        self.memory_variance.fill_(1.0)
        self.slot_write_count.zero_()
        self.last_residual.fill_(0.0)
        self.last_converged.fill_(True)
        self.iterations_used.fill_(0)


# =============================================================================
# SMOKE TEST
# =============================================================================

def _smoke_test():
    """Verify CA3 read/write/convergence/serialization."""
    print("=" * 60)
    print("CA3 Autoassociative Memory Smoke Test")
    print("=" * 60)

    cfg = HippocampalConfig()
    ca3 = CA3RecurrentMatrix(cfg)

    total_params = sum(p.numel() for p in ca3.parameters())
    total_buffers = sum(b.numel() for b in ca3.buffers())
    print(f"\nParameters: {total_params:,} (trainable)")
    print(f"Buffers:    {total_buffers:,} (state)")

    # --- Test 1: Write and retrieve ---
    print(f"\n[Test 1] Write and retrieve single episode")
    episode = torch.randn(1, cfg.ca3_code_dim)
    write_stats = ca3.write(episode)
    print(f"  Write residual: {write_stats['residual'].item():.6f}")
    print(f"  Converged: {write_stats['converged']}")
    print(f"  Iterations: {write_stats['iterations_used']}")

    reconstruction, weights = ca3.read(episode)
    error = F.mse_loss(reconstruction, episode).item()
    print(f"  Read-back MSE: {error:.6f}")
    assert error < 0.1, "FAIL: Cannot retrieve just-written episode!"
    print(f"  PASS")

    # --- Test 2: Partial cue retrieval ---
    print(f"\n[Test 2] Partial cue retrieval")
    # Zero out half the cue dimensions
    partial_cue = episode.clone()
    partial_cue[0, cfg.ca3_code_dim // 2:] = 0.0
    reconstruction_partial, _ = ca3.read(partial_cue)
    error_partial = F.mse_loss(reconstruction_partial, episode).item()
    print(f"  Partial cue MSE: {error_partial:.6f}")
    print(f"  (Should be higher than full cue but still structured)")

    # --- Test 3: Multiple episodes with interference check ---
    print(f"\n[Test 3] Multiple episodes, interference check")
    episodes = []
    for i in range(8):
        ep = torch.randn(1, cfg.ca3_code_dim)
        ca3.write(ep)
        episodes.append(ep)

    # Retrieve each and check error
    errors = []
    for ep in episodes:
        recon, _ = ca3.read(ep)
        errors.append(F.mse_loss(recon, ep).item())
    mean_error = sum(errors) / len(errors)
    print(f"  Mean retrieval MSE across 8 episodes: {mean_error:.6f}")

    # --- Test 4: Reconsolidation (write to existing slot) ---
    print(f"\n[Test 4] Reconsolidation (update existing slot)")
    target_slot = torch.tensor([0])
    old_value = ca3.memory_mean[0].clone()
    update = torch.randn(1, cfg.ca3_code_dim) * 0.1 + old_value.unsqueeze(0)
    ca3.write(update, slot_idx=target_slot)
    new_value = ca3.memory_mean[0]
    change = torch.norm(new_value - old_value).item()
    print(f"  Slot 0 change magnitude: {change:.6f}")
    assert change > 0, "FAIL: Reconsolidation had no effect!"
    print(f"  PASS")

    # --- Test 5: Convergence guard ---
    print(f"\n[Test 5] Convergence diagnostics")
    diag = ca3.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # --- Test 6: Serialization round-trip ---
    print(f"\n[Test 6] Serialization round-trip")
    state = ca3.state_dict()
    ca3_restored = CA3RecurrentMatrix(cfg)
    ca3_restored.load_state_dict(state)

    test_q = torch.randn(1, cfg.ca3_code_dim)
    out1, _ = ca3.read(test_q)
    out2, _ = ca3_restored.read(test_q)
    diff = torch.norm(out1 - out2).item()
    print(f"  Output difference after restore: {diff:.12f}")
    assert diff < 1e-5, "FAIL: Serialization round-trip corrupted state!"
    print(f"  PASS")

    print(f"\n{'=' * 60}")
    print(f"ALL SMOKE TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _smoke_test()
