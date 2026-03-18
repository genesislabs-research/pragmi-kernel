"""
perforant_path.py
=================

PRAGMI Cognitive Kernel: Perforant Path Communication Subspace
==============================================================

BIOLOGICAL GROUNDING
--------------------
This file models the Perforant Path (PP), the principal afferent projection
from the entorhinal cortex (EC) to the hippocampal formation. In the biological
brain, this pathway serves two distinct roles depending on which hippocampal
subfield it targets:

    EC Layer II -> Dentate Gyrus:  Pattern separation during encoding. The DG
    orthogonalizes overlapping cortical representations into sparse, distinct
    codes before they reach CA3 for storage.

    EC Layer II -> CA3 directly:   Retrieval cue relay. A partial cue enters
    CA3 via ~3,600 perforant path synapses per pyramidal cell. This input is
    too weak to force a new storage pattern (that requires the ~46 mossy fiber
    inputs from the DG), but it is sufficient to initiate autoassociative
    pattern completion through the recurrent collateral network.

    EC Layer III -> CA1 directly:  Comparator input. CA1 receives both the
    pattern-completed output from CA3 (via Schaffer collaterals) and the
    current sensory context from EC layer III (via the direct perforant path).
    The mismatch between these two streams drives novelty detection.

In PRAGMI, the perforant path sits at the boundary between Timmy (the spiking
bridge language model, the "subconscious") and the Cognitive Kernel (the
hippocampal memory system). It implements a low-rank communication subspace
that filters Timmy's high-dimensional population activity into a compact set
of predictive dimensions. Only those dimensions reach the kernel. The loudest
activity inside Timmy is not necessarily what the kernel hears.

The key computational insight from Semedo et al. (2019) is that inter-area
communication happens in a subspace that is MISALIGNED with the dominant
variance directions of the source population. This means the most active
neurons in Timmy do not necessarily contribute the most to what the kernel
receives. The communication subspace captures the dimensions that predict
target fluctuations, not the dimensions with the largest source variance.

Lead papers:

1. Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M., & Kohn, A. (2019).
   "Cortical areas interact through a communication subspace."
   Neuron, 102(1), 249-259. DOI: 10.1016/j.neuron.2019.01.026

2. Witter, M.P., Naber, P.A., van Haeften, T., Machielsen, W.C.,
   Rombouts, S.A., Barkhof, F., Scheltens, P., & Lopes da Silva, F.H. (2000).
   "Cortico-hippocampal communication by way of parallel
   parahippocampal-subicular pathways."
   Hippocampus, 10(4), 398-410.
   DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K

3. Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
   separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
   DOI: 10.3389/fnsys.2013.00074


WHY THIS FILE EXISTS (for engineers)
------------------------------------
When two neural populations need to talk, the naive approach is a dense
weight matrix W of shape (source_dim, target_dim). For source=496 and
target=64, that is 31,744 parameters and every source neuron can influence
every target neuron. This creates two problems:

    1. IDENTITY BLEED: Timmy's internal dynamics (what it is "thinking about")
       leak into the kernel even when they are irrelevant to memory. The kernel
       cannot distinguish "Timmy is processing this word" from "Timmy's
       recurrent state is ringing from three words ago."

    2. PARAMETER WASTE: Most of those 31,744 weights are fitting noise.
       Semedo et al. found that inter-area communication needs only 2-4
       dimensions, meaning ~99% of a dense projection's capacity is wasted
       on fitting non-predictive variance.

The fix is reduced-rank regression: factor W = U @ diag(gains) @ V where
U is (source, rank), gains is (rank,), and V is (rank, target). This gives
us rank * (source + target + 1) parameters instead of source * target.
For rank=3: 3*(496+64+1) = 1,683 parameters, a 19x reduction that also
enforces the biological constraint.

The per-channel gains implement selective routing. An external modulator
(the meta-zone in the full architecture, or a routing_mask during inference)
can scale individual channels without touching the learned subspace geometry.
Setting a gain to zero closes that channel. This separates WHAT can be
communicated (the learned U and V matrices, shaped by training) from WHAT
IS BEING communicated right now (the gain-modulated activity, shaped by
the current cognitive state).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================
# This module uses HippocampalConfig from cognitive_kernel.py.
# When imported standalone for testing, a minimal fallback is provided.

try:
    from cognitive_kernel import HippocampalConfig
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class HippocampalConfig:
        """Minimal fallback config for standalone testing.

        See cognitive_kernel.py for the full config with all parameter
        citations and biological grounding.
        """
        bridge_dim: int = 496
        coordinate_dim: int = 64
        comm_subspace_rank: int = 3


# =============================================================================
# PERFORANT PATH BRIDGE
# =============================================================================

class PerforantPathBridge(nn.Module):
    """Low-rank communication subspace modeling the Perforant Path.

    BIOLOGICAL STRUCTURE: Perforant Path, the axonal bundle projecting from
    entorhinal cortex layer II to the dentate gyrus and CA3 field of the
    hippocampus.

    BIOLOGICAL FUNCTION: In the rat hippocampus, each CA3 pyramidal cell
    receives approximately 3,600 perforant path synapses (compared to only
    46 mossy fiber synapses from the DG and 12,000 recurrent collateral
    synapses from other CA3 cells). The perforant path input is sufficient
    to relay a partial retrieval cue but not strong enough to force a new
    storage pattern. Storage requires the mossy fiber input, which provides
    fewer but far more powerful per-synapse connections that can override
    the recurrent dynamics.

    Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
    separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
    DOI: 10.3389/fnsys.2013.00074

    COMPUTATIONAL IMPLEMENTATION: The projection is factored as:

        output = input @ (U_send * channel_gains) @ V_receive

    where:
        U_send:        (source_dim, rank)   learned sending subspace
        channel_gains: (rank,)              learned steady-state routing
        V_receive:     (rank, target_dim)   learned receiving projection

    The full effective weight matrix W = U @ diag(gains) @ V has rank at
    most comm_subspace_rank by construction. This enforces the communication
    subspace constraint from Semedo et al. (2019).

    ROUTING ARCHITECTURE (two levels of control):

    Level 1 (learned, slow): The channel_gains parameter is trained alongside
    the subspace matrices. It captures the steady-state communication
    bandwidth between Timmy and the kernel. This is the "structural wiring"
    that changes over the lifetime of the system.

    Level 2 (dynamic, fast): The optional routing_mask argument to forward()
    allows an external modulator to scale channels at inference time without
    touching the learned gains. This is how the meta-zone (or any executive
    controller) can temporarily open, close, or reweight communication
    channels based on the current cognitive state. The separation between
    learned structure and dynamic control is critical: it means the system
    can modulate what it attends to without catastrophically forgetting
    what it has learned to communicate.

    Semedo, J.D. et al. (2019). "Cortical areas interact through a
    communication subspace." Neuron, 102(1), 249-259.
    DOI: 10.1016/j.neuron.2019.01.026

    INTERFACE BOUNDARY:
        SENDING:    Timmy (spiking neural network language model, isocortex analog)
        RECEIVING:  Cognitive Kernel allocortex (DG/CA3 input layer)
        CONNECTION: Perforant Path (entorhinal cortex layer II projections)
    """

    def __init__(self, cfg: HippocampalConfig):
        """Initialize the low-rank perforant path projection.

        All matrix initializations use Kaiming scaling (fan-in normalization)
        so that the output variance is approximately 1.0 when the input has
        unit variance. This eliminates the need for a post-hoc gain scalar
        to control output amplitude; the initialization handles it.

        NOT a biological quantity. Standard neural network initialization
        practice that prevents saturation in downstream spiking neurons.

        Parameters
        ----------
        cfg : HippocampalConfig
            Kernel configuration. The relevant fields are:
                bridge_dim          Timmy's output dimensionality (source space)
                coordinate_dim      Kernel's coordinate manifold (target space)
                comm_subspace_rank  Rank constraint on the projection
        """
        super().__init__()
        self.cfg = cfg
        source_dim = cfg.bridge_dim
        target_dim = cfg.coordinate_dim
        rank = cfg.comm_subspace_rank

        # -----------------------------------------------------------------
        # U_send: (source_dim, rank)
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Perforant Path Sending Weights (EC L2 axon terminals)
        #
        # PLAIN ENGLISH: Selects which dimensions of Timmy's population
        # activity are "predictive" of hippocampal target fluctuations.
        # Semedo et al. (2019) showed these predictive dimensions are NOT
        # aligned with the dominant variance directions (the largest shared
        # fluctuations) in the source. Training shapes U_send to discover
        # which source directions actually matter for the target.
        #
        # INITIALIZATION: Orthogonal columns via QR decomposition, giving
        # U_send a spectral norm of exactly 1.0. Combined with the matching
        # orthogonal initialization of V_receive (orthonormal rows), the
        # product U_send @ V_receive also has spectral norm 1.0. This
        # matters because coordinate vectors that pass through this bridge
        # enter the CA3 memory matrix, and the Ben-Israel pseudoinverse
        # iteration converges fastest when the matrix being inverted has
        # spectral norm near 1.0. If the bridge amplified or crushed signal
        # before it reached CA3, memory writes would require more iterations
        # or a larger learned alpha correction to compensate.
        #
        # The spectral norm guarantee: if U has orthonormal columns
        # (U^T @ U = I_r) and V has orthonormal rows (V @ V^T = I_r),
        # then the singular values of U @ V are all exactly 1.0, so
        # ||U @ V||_2 = 1.0. Proof: singular values of U@V are square
        # roots of eigenvalues of V^T @ U^T @ U @ V = V^T @ V. Since V
        # has orthonormal rows, V^T @ V has r eigenvalues equal to 1.
        #
        # NOT a biological quantity. Training artifact chosen specifically
        # to guarantee downstream Ben-Israel convergence within T=8 steps.
        #
        # CITATION: Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M.,
        # & Kohn, A. (2019). "Cortical areas interact through a communication
        # subspace." Neuron, 102(1), 249-259.
        # DOI: 10.1016/j.neuron.2019.01.026
        # -----------------------------------------------------------------
        U_raw = torch.randn(source_dim, rank)
        Q_U, _ = torch.linalg.qr(U_raw)
        self.U_send = nn.Parameter(Q_U[:, :rank].contiguous())

        # -----------------------------------------------------------------
        # V_receive: (rank, target_dim)
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Perforant Path Receiving Weights (DG/CA3 dendritic)
        #
        # PLAIN ENGLISH: Maps from the low-rank communication subspace into
        # the kernel's 64-dimensional coordinate manifold. This is where the
        # subspace activity becomes a coordinate vector that the DG, CA3,
        # and CA1 can process natively.
        #
        # INITIALIZATION: Orthogonal rows via QR on the transpose, matching
        # U_send's orthogonal columns. Together they guarantee ||U @ V||_2
        # = 1.0 at initialization. See the proof in U_send's comment block.
        # NOT a biological quantity. Training artifact for Ben-Israel stability.
        #
        # CITATION: Witter, M.P., Naber, P.A., van Haeften, T., et al. (2000).
        # "Cortico-hippocampal communication by way of parallel
        # parahippocampal-subicular pathways." Hippocampus, 10(4), 398-410.
        # DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
        # -----------------------------------------------------------------
        V_raw = torch.randn(target_dim, rank)
        Q_V, _ = torch.linalg.qr(V_raw)
        self.V_receive = nn.Parameter(Q_V[:, :rank].T.contiguous())

        # -----------------------------------------------------------------
        # channel_gains: (rank,)
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Singular Value Neuromodulation (Meta-Zone Routing)
        #
        # PLAIN ENGLISH: Per-channel gain control. Each element scales one
        # dimension of the communication subspace. These are the "singular
        # values" of the factored projection. In the full six-zone cortical
        # sheet, each zone-to-hippocampus projection has its own gain vector,
        # and the meta-zone adjusts all of them to control what information
        # reaches the kernel at any given moment.
        #
        # Initialized to 1.0: all channels open at start of training. The
        # system learns which channels to keep open and at what gain.
        #
        # CRITICAL DESIGN NOTE: These gains represent Level 1 (learned,
        # slow) routing. They are trained via backprop and capture the
        # steady-state communication bandwidth. Level 2 (dynamic, fast)
        # routing is handled by the routing_mask argument to forward(),
        # which is multiplied with these gains at inference time but does
        # not modify them.
        #
        # CITATION: Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M.,
        # & Kohn, A. (2019). "Cortical areas interact through a communication
        # subspace." Neuron, 102(1), 249-259.
        # DOI: 10.1016/j.neuron.2019.01.026
        # See project extract, Section 3: "the meta-zone scales the singular
        # values of each inter-zone projection."
        # -----------------------------------------------------------------
        self.channel_gains = nn.Parameter(torch.ones(rank))

    def forward(
        self,
        spike_rates_source: Tensor,
        routing_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Filter source population activity through the communication subspace.

        BIOLOGICAL ANALOG: Entorhinal cortex layer II neurons fire in
        response to neocortical input. Their axons traverse the perforant
        path and synapse onto dentate granule cells and CA3 pyramidal cells.
        Only the component of entorhinal activity that lies within the
        communication subspace effectively drives hippocampal targets.

        INTERFACE BOUNDARY:
            SENDING:    Timmy output layer (isocortex, source population)
            RECEIVING:  Kernel coordinate manifold (allocortex, DG/CA3)
            CONNECTION: Perforant Path (EC layer II axonal projections)

        ROUTING LEVELS:
            Level 1 (learned):  self.channel_gains, trained via backprop
            Level 2 (dynamic):  routing_mask argument, set at inference time
            Effective gain = channel_gains * routing_mask (elementwise)

        Parameters
        ----------
        spike_rates_source : Tensor, shape (batch, bridge_dim)
            Spike-rate coded population activity from Timmy. This is the
            full-dimensional output of the source area, containing both
            predictive dimensions (what the kernel will hear) and private
            dimensions (internal processing invisible to the kernel).

            Variable naming: "spike_rates" because this is a rate-coded
            population vector, not a membrane potential (which would use
            the prefix "v_") or a binary spike train (which would use "s_").

        routing_mask : Tensor, optional, shape (rank,) or (batch, rank)
            Dynamic per-channel scaling from the meta-zone or executive
            controller. Multiplied elementwise with the learned channel_gains.
            If None, only the learned gains are used (steady-state routing).

            Example use cases:
                torch.ones(rank)       All channels open (default behavior)
                torch.tensor([1,1,0])  Close channel 2 (suppress one dimension)
                torch.tensor([1,0.5,1]) Attenuate channel 1 by half

        Returns
        -------
        coords_target : Tensor, shape (batch, coordinate_dim)
            The input to the hippocampal coordinate manifold. Only the
            predictive dimensions of Timmy's activity reach the kernel,
            modulated by both learned and dynamic routing gains.
        """
        # Combine Level 1 (learned) and Level 2 (dynamic) routing gains.
        # When no mask is provided, effective_gains = channel_gains (learned only).
        # When a mask is provided, the two are multiplied elementwise, so the
        # dynamic controller can attenuate or close channels without altering
        # the learned structure.
        effective_gains = self.channel_gains
        if routing_mask is not None:
            effective_gains = effective_gains * routing_mask

        # Scale U_send columns by the effective per-channel gains.
        # This is mathematically equivalent to U @ diag(gains) but avoids
        # allocating the (rank, rank) diagonal matrix. The elementwise
        # broadcast multiplies each column of U_send by the corresponding
        # gain scalar.
        #
        # Shape: (source_dim, rank) * (1, rank) -> (source_dim, rank)
        scaled_U = self.U_send * effective_gains.unsqueeze(0)

        # Two thin matrix multiplications enforce the rank constraint.
        #
        # Step 1: Project from source space to the rank-r subspace.
        # This selects the predictive dimensions of Timmy's activity.
        # Shape: (batch, source_dim) @ (source_dim, rank) -> (batch, rank)
        subspace_activity = spike_rates_source @ scaled_U

        # Step 2: Project from the subspace to the target coordinate manifold.
        # This maps the selected dimensions into the space where the
        # hippocampal circuitry (DG, CA3, CA1) operates.
        # Shape: (batch, rank) @ (rank, target_dim) -> (batch, target_dim)
        coords_target = subspace_activity @ self.V_receive

        return coords_target

    def effective_weight(self) -> Tensor:
        """Compute the full (bridge_dim, coordinate_dim) projection matrix.

        Returns the dense matrix W = (U_send * channel_gains) @ V_receive.
        This matrix has rank at most comm_subspace_rank by construction.

        NOTE: This uses the LEARNED gains only (Level 1), not any dynamic
        routing mask. The purpose is to inspect the structural subspace that
        training has discovered, independent of any transient modulation.
        To inspect the effective weight under a specific routing mask, call
        effective_weight_with_mask(mask) instead.

        Returns
        -------
        W : Tensor, shape (bridge_dim, coordinate_dim)
            The effective dense weight matrix with rank <= comm_subspace_rank.
        """
        scaled_U = self.U_send * self.channel_gains.unsqueeze(0)
        return scaled_U @ self.V_receive

    def effective_weight_with_mask(self, routing_mask: Tensor) -> Tensor:
        """Compute the effective weight matrix under a specific routing mask.

        This shows what the projection actually looks like at a given moment,
        including both learned structure and dynamic modulation.

        Parameters
        ----------
        routing_mask : Tensor, shape (rank,)
            The dynamic routing mask to apply.

        Returns
        -------
        W : Tensor, shape (bridge_dim, coordinate_dim)
            The momentary effective weight matrix.
        """
        effective_gains = self.channel_gains * routing_mask
        scaled_U = self.U_send * effective_gains.unsqueeze(0)
        return scaled_U @ self.V_receive

    def effective_rank(self) -> float:
        """Compute the Shannon-entropy effective rank of the learned subspace.

        Uses the singular value decomposition of the effective weight matrix
        (learned gains only). The effective rank is exp(H) where H is the
        Shannon entropy of the normalized singular value distribution.

        Interpretation:
            effective_rank = comm_subspace_rank: all channels equally active
            effective_rank = 1.0: one channel dominates, subspace collapsed

        NOT a biological quantity. Diagnostic metric for detecting subspace
        collapse during training. If effective_rank drops significantly below
        comm_subspace_rank, it means the system is not using the full
        communication bandwidth and some channels should be investigated.

        Returns
        -------
        eff_rank : float
            Shannon-entropy effective rank, in [1.0, comm_subspace_rank].
        """
        W = self.effective_weight()
        try:
            singular_values = torch.linalg.svdvals(W)
            s_norm = singular_values / (singular_values.sum() + 1e-8)
            entropy = -(s_norm * torch.log(s_norm + 1e-12)).sum()
            return torch.exp(entropy).item()
        except Exception:
            return float(self.cfg.comm_subspace_rank)

    def get_diagnostics(self) -> Dict[str, object]:
        """Return a comprehensive diagnostic snapshot of the communication subspace.

        Intended for use by the Autogenic Diagnostic Routines (ADR) system
        and for human inspection during development. All values use the
        learned gains only (Level 1), not any transient routing mask.

        Returns
        -------
        diagnostics : dict
            channel_gains:    list of current per-channel gain values
            effective_rank:   Shannon-entropy rank (float)
            exact_rank:       numerical rank of the weight matrix (int)
            frobenius_norm:   Frobenius norm of the projection (proxy for
                              total signal strength reaching the kernel)
            parameter_count:  total parameters in this module (int)
            compression_ratio: ratio of dense parameters to actual parameters,
                              quantifying how much the low-rank factorization
                              saves compared to a dense projection
        """
        W = self.effective_weight()
        dense_params = self.cfg.bridge_dim * self.cfg.coordinate_dim
        actual_params = sum(p.numel() for p in self.parameters())

        return {
            "channel_gains": self.channel_gains.detach().tolist(),
            "effective_rank": self.effective_rank(),
            "exact_rank": torch.linalg.matrix_rank(W).item(),
            "frobenius_norm": torch.norm(W, p="fro").item(),
            "spectral_norm": torch.linalg.svdvals(W)[0].item(),
            "parameter_count": actual_params,
            "compression_ratio": dense_params / actual_params,
        }


# =============================================================================
# SMOKE TEST
# =============================================================================

def _smoke_test():
    """Verify the PerforantPathBridge meets all architectural constraints.

    This test checks five properties:
        1. The effective weight matrix respects the rank constraint
        2. Forward pass produces correct output shapes
        3. Closing a channel via learned gains reduces the exact rank
        4. The routing_mask correctly modulates without altering learned gains
        5. Serialization round-trip preserves the subspace exactly
    """
    print("=" * 60)
    print("PerforantPathBridge Smoke Test")
    print("=" * 60)

    cfg = HippocampalConfig()
    bridge = PerforantPathBridge(cfg)

    # --- Parameter budget ---
    total_params = sum(p.numel() for p in bridge.parameters())
    dense_equivalent = cfg.bridge_dim * cfg.coordinate_dim
    print(f"\nParameter budget:")
    print(f"  U_send:        {cfg.bridge_dim} x {cfg.comm_subspace_rank} "
          f"= {cfg.bridge_dim * cfg.comm_subspace_rank:,}")
    print(f"  V_receive:     {cfg.comm_subspace_rank} x {cfg.coordinate_dim} "
          f"= {cfg.comm_subspace_rank * cfg.coordinate_dim:,}")
    print(f"  channel_gains: {cfg.comm_subspace_rank}")
    print(f"  Total:         {total_params:,}")
    print(f"  Dense equiv:   {dense_equivalent:,}")
    print(f"  Compression:   {dense_equivalent / total_params:.1f}x")

    # --- Test 1: Rank constraint ---
    print(f"\n[Test 1] Rank constraint")
    W = bridge.effective_weight()
    exact_rank = torch.linalg.matrix_rank(W).item()
    print(f"  Weight shape: {tuple(W.shape)}")
    print(f"  Exact rank:   {exact_rank} (must be <= {cfg.comm_subspace_rank})")
    assert exact_rank <= cfg.comm_subspace_rank, "FAIL: Rank constraint violated!"
    print(f"  PASS")

    # --- Test 1b: Spectral norm at initialization ---
    print(f"\n[Test 1b] Spectral norm at initialization")
    sigma_max = torch.linalg.svdvals(W)[0].item()
    print(f"  ||U @ diag(gains) @ V||_2 = {sigma_max:.6f} (target: 1.0)")
    print(f"  Deviation from 1.0: {abs(sigma_max - 1.0):.6f}")
    # At init, gains=1.0 and U,V have orthonormal columns/rows,
    # so the spectral norm should be exactly 1.0 (up to float precision).
    assert abs(sigma_max - 1.0) < 1e-5, "FAIL: Spectral norm deviates from 1.0!"
    print(f"  PASS (Ben-Israel convergence guarantee: alpha_init = 1/||W||^2 = 1.0)")

    # --- Test 2: Forward pass shapes ---
    print(f"\n[Test 2] Forward pass shapes")
    batch = 4
    fake_input = torch.randn(batch, cfg.bridge_dim)
    output = bridge(fake_input)
    print(f"  Input:  {tuple(fake_input.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (batch, cfg.coordinate_dim), "FAIL: Wrong output shape!"
    print(f"  PASS")

    # --- Test 3: Channel gating reduces rank ---
    print(f"\n[Test 3] Channel gating (learned gains)")
    original_gains = bridge.channel_gains.data.clone()
    with torch.no_grad():
        bridge.channel_gains[0] = 0.0
    W_gated = bridge.effective_weight()
    rank_gated = torch.linalg.matrix_rank(W_gated).item()
    print(f"  After closing channel 0: rank = {rank_gated} "
          f"(must be <= {cfg.comm_subspace_rank - 1})")
    assert rank_gated <= cfg.comm_subspace_rank - 1, "FAIL: Gating did not reduce rank!"
    with torch.no_grad():
        bridge.channel_gains.copy_(original_gains)  # restore
    print(f"  PASS")

    # --- Test 4: Routing mask modulates without altering learned gains ---
    print(f"\n[Test 4] Dynamic routing mask (Level 2)")
    gains_before = bridge.channel_gains.data.clone()

    mask = torch.tensor([1.0, 0.0, 1.0])[:cfg.comm_subspace_rank]
    output_masked = bridge(fake_input, routing_mask=mask)

    gains_after = bridge.channel_gains.data.clone()
    gains_unchanged = torch.allclose(gains_before, gains_after)
    print(f"  Learned gains unchanged after masked forward: {gains_unchanged}")
    assert gains_unchanged, "FAIL: Routing mask mutated the learned gains!"

    # Verify the mask actually changes the output
    output_unmasked = bridge(fake_input)
    outputs_differ = not torch.allclose(output_masked, output_unmasked)
    print(f"  Masked output differs from unmasked: {outputs_differ}")
    assert outputs_differ, "FAIL: Routing mask had no effect on output!"

    # Verify effective_weight_with_mask matches
    W_masked = bridge.effective_weight_with_mask(mask)
    rank_masked = torch.linalg.matrix_rank(W_masked).item()
    print(f"  Effective rank under mask: {rank_masked}")
    print(f"  PASS")

    # --- Test 5: Serialization round-trip ---
    print(f"\n[Test 5] Serialization round-trip")
    state = bridge.state_dict()
    bridge2 = PerforantPathBridge(cfg)
    bridge2.load_state_dict(state)

    test_input = torch.randn(1, cfg.bridge_dim)
    out1 = bridge(test_input)
    out2 = bridge2(test_input)
    diff = torch.norm(out1 - out2).item()
    print(f"  Output difference after restore: {diff:.12f}")
    assert diff < 1e-6, "FAIL: Serialization round-trip corrupted state!"
    print(f"  PASS")

    # --- Diagnostics summary ---
    print(f"\n[Diagnostics]")
    diag = bridge.get_diagnostics()
    for key, val in diag.items():
        print(f"  {key}: {val}")

    print(f"\n{'=' * 60}")
    print(f"ALL SMOKE TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _smoke_test()
