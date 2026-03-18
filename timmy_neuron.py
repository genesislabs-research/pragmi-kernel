"""
timmy/neuron.py
===============

PRAGMI Spiking Bridge (Timmy): Core Neuron Model
=================================================

BIOLOGICAL GROUNDING
--------------------
This file implements the fundamental spiking neuron used throughout Timmy,
the spiking neural network language model that sits between the external LLM
and the Cognitive Kernel. Every neuron in Timmy's sensory, association, and
executive zones is an instance of the AssociativeLIF class defined here.

The neuron model is a Leaky Integrate-and-Fire (LIF) neuron with three
extensions beyond the basic textbook LIF:

    1. SYNAPTIC CURRENT FILTERING: Input current passes through a first-order
       low-pass filter (exponential decay with time constant tau_syn) before
       reaching the membrane. This models the postsynaptic current dynamics
       caused by neurotransmitter binding and ion channel kinetics. Without
       this filter, input spikes would produce instantaneous voltage jumps,
       which is physically unrealistic.

    2. ABSOLUTE REFRACTORY PERIOD: After firing, the neuron is clamped to
       the reset voltage for a fixed number of timesteps. During this period,
       no input can trigger another spike. This models the inactivation of
       sodium channels immediately after an action potential, during which
       the neuron is physically unable to fire regardless of input strength.

    3. CASCADE AMPLIFICATION: When a neuron fires, it boosts the synaptic
       current of nearby neurons in the same cortical minicolumn. This models
       the lateral excitatory connections within a minicolumn that produce
       correlated bursting. It gives the network population-level dynamics
       beyond what independent LIF neurons can produce.

The neuron's spike function uses a surrogate gradient (arctangent) for
backpropagation, since the true Heaviside step function has zero gradient
everywhere except at the threshold, where it is undefined.

Lead papers:

1. Neftci, E.O., Mostafa, H., & Zenke, F. (2019). "Surrogate gradient
   learning in spiking neural networks: Bringing the power of gradient-based
   optimization to spiking neural networks." IEEE Signal Processing Magazine,
   36(6), 51-63. DOI: 10.1109/MSP.2019.2931595

2. Gerstner, W., Kistler, W.M., Naud, R., & Paninski, L. (2014). "Neuronal
   Dynamics: From single neurons to networks and models of cognition."
   Cambridge University Press. DOI: 10.1017/CBO9781107447615

3. Mountcastle, V.B. (1997). "The columnar organization of the neocortex."
   Brain, 120(4), 701-722. DOI: 10.1093/brain/120.4.701


WHY THIS FILE EXISTS (for engineers)
------------------------------------
A spiking neural network processes information through discrete binary events
(spikes) rather than continuous activations. Each neuron integrates incoming
current over time, and when its membrane potential crosses a threshold, it
emits a spike (output = 1.0) and resets. Between spikes, the membrane leaks
back toward its resting potential. This temporal dynamics is what gives SNNs
their computational properties: spike timing carries information, and the
membrane acts as a natural temporal integrator.

The engineering challenge is that the spike function (a Heaviside step) has
zero gradient almost everywhere, so standard backpropagation cannot train
the network. The surrogate gradient solves this by replacing the Heaviside's
gradient with a smooth approximation (the derivative of arctan) during the
backward pass only. The forward pass still produces hard binary spikes. This
is the standard approach in modern SNN training and is not specific to our
architecture.

The cascade amplification is specific to our architecture. Standard LIF
neurons are independent: one neuron's spike does not directly affect its
neighbors' membrane potential. We add local lateral excitation within
clusters of neurons (modeling cortical minicolumns), so that a spike in one
neuron boosts the synaptic current in nearby neurons. This produces
correlated population bursting that carries more information per timestep
than independent spikes.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NeuronConfig:
    """Configuration for the spiking neuron model.

    All biologically derived parameters cite their source. Parameters that
    are training artifacts or engineering approximations are labeled explicitly.
    """

    # --- Membrane dynamics ---

    tau_mem: float = 0.85
    """Membrane decay factor (discrete-time). The membrane potential at each
    timestep is: v_new = tau_mem * v_old + (1 - tau_mem) * i_syn.
    In continuous time, the membrane time constant of cortical pyramidal
    cells is typically 10-30ms. In discrete time with a 1ms step,
    beta = exp(-dt/tau) gives values around 0.90-0.97 for that range. Our
    value of 0.85 corresponds to a faster-decaying neuron (~6ms effective
    time constant), which was found empirically to produce stable training
    dynamics in the SNN LM at T=8 timesteps.
    NOT a biological quantity. Training artifact tuned for LM performance.
    Gerstner, W. et al. (2014). DOI: 10.1017/CBO9781107447615 (Chapter 1)"""

    tau_mem_min: float = 0.8
    """Lower clamp on learnable membrane decay. Prevents the neuron from
    becoming so leaky that it cannot integrate over even two timesteps.
    NOT a biological quantity. Stability guard."""

    tau_mem_max: float = 0.98
    """Upper clamp on learnable membrane decay. Prevents the neuron from
    becoming a perfect integrator (no leak), which would cause unbounded
    membrane potential growth.
    NOT a biological quantity. Stability guard."""

    tau_syn: float = 0.50
    """Synaptic current decay factor (discrete-time). Models the first-order
    dynamics of postsynaptic current: i_syn_new = tau_syn * i_syn_old + input.
    A value of 0.50 means the synaptic current halves every timestep, which
    corresponds to a fast AMPA-like synapse (~2ms decay at 1ms steps).
    Biological AMPA receptor time constants are 1-5ms; NMDA are 50-150ms.
    NOT a biological quantity. Training artifact biased toward fast dynamics.
    Gerstner, W. et al. (2014). DOI: 10.1017/CBO9781107447615 (Chapter 3)"""

    v_threshold: float = 0.12
    """Initial spike threshold (learnable, per-neuron). When the membrane
    potential v_mem exceeds this value, the neuron fires a spike. The
    threshold is learned during training and clamped to [v_thresh_min,
    v_thresh_max]. Biological spike thresholds are approximately -55mV to
    -40mV (relative to a resting potential of -70mV), but our membrane
    operates in arbitrary units, not millivolts.
    NOT a biological quantity. The absolute value is meaningless; only its
    relationship to the input scale matters.
    Gerstner, W. et al. (2014). DOI: 10.1017/CBO9781107447615 (Chapter 1)"""

    v_thresh_min: float = 0.05
    """Lower clamp on learnable threshold. Prevents the threshold from
    reaching zero (where every input would trigger a spike).
    NOT a biological quantity. Stability guard."""

    v_thresh_max: float = 0.5
    """Upper clamp on learnable threshold. Prevents the threshold from
    becoming so high that the neuron never fires.
    NOT a biological quantity. Stability guard."""

    v_reset: float = -0.1
    """Membrane potential during the refractory period. After a spike, the
    membrane is clamped to this value for refractory_t timesteps. The
    negative value models the afterhyperpolarization (AHP) caused by
    potassium channel activation following an action potential.
    Gerstner, W. et al. (2014). DOI: 10.1017/CBO9781107447615 (Chapter 1)"""

    refractory_t: int = 2
    """Duration of the absolute refractory period in timesteps. During this
    period, the neuron cannot fire regardless of input. The biological
    absolute refractory period for cortical pyramidal cells is approximately
    1-2ms. With T=8 timesteps, a refractory period of 2 timesteps limits
    the maximum firing rate to T/(refractory_t+1) = 2.67 spikes per window.
    Hodgkin, A.L. & Huxley, A.F. (1952). "A quantitative description of
    membrane current and its application to conduction and excitation in
    nerve." Journal of Physiology, 117(4), 500-544.
    DOI: 10.1113/jphysiol.1952.sp004764"""

    # --- Surrogate gradient ---

    surrogate_alpha: float = 4.0
    """Sharpness parameter for the arctangent surrogate gradient. Higher
    values make the surrogate closer to the true Heaviside (sharper
    transition), but reduce gradient magnitude far from the threshold.
    Lower values produce smoother gradients but less faithful spike dynamics.
    The value 4.0 was found by Fang et al. (2021) to work well for deep
    SNN training.
    NOT a biological quantity. Training artifact only. The biological spike
    mechanism has no "gradient" in the backpropagation sense.
    Neftci, E.O., Mostafa, H., & Zenke, F. (2019). "Surrogate gradient
    learning in spiking neural networks." IEEE Signal Processing Magazine,
    36(6), 51-63. DOI: 10.1109/MSP.2019.2931595"""

    # --- Cascade amplification ---

    n_clusters: int = 64
    """Number of neuron clusters (minicolumn analogs). Neurons are assigned
    to clusters in round-robin order: neuron i belongs to cluster (i mod
    n_clusters). Each cluster models a cortical minicolumn, a vertical
    column of ~80-100 neurons that share similar tuning properties and
    have strong lateral excitatory connections.
    Mountcastle, V.B. (1997). "The columnar organization of the neocortex."
    Brain, 120(4), 701-722. DOI: 10.1093/brain/120.4.701"""

    cascade_radius: int = 3
    """Lateral excitation radius in cluster index space. A spike in cluster
    i excites clusters (i-r) through (i+r) with strength decaying linearly
    with distance. This models the lateral spread of excitation within and
    between adjacent minicolumns.
    NOT a biological quantity. The spatial extent of lateral excitation in
    real cortex depends on axonal arbor size (~200-500um), which does not
    map directly to an integer cluster index."""

    cascade_gain: float = 0.8
    """Initial gain for cascade amplification. Controls how much a spike
    in one cluster boosts the synaptic current in neighboring clusters.
    Learnable per-cluster during training.
    NOT a biological quantity. Engineering approximation of lateral
    excitatory coupling strength."""

    # --- Homeostatic monitoring ---

    target_spike_rate: float = 0.05
    """Target firing rate for the exponential moving average tracker.
    Used by the AuxiliarySpikeRegulator (in blocks.py) to penalize neurons
    that fire too much or too little.
    NOT a biological quantity. Training regularization target."""


# =============================================================================
# SURROGATE GRADIENT
# =============================================================================

class ATanSurrogate(torch.autograd.Function):
    """Arctangent surrogate gradient for spiking neuron training.

    BIOLOGICAL NAME: none (there is no biological analog to backpropagation)

    PLAIN ENGLISH: Biological neurons fire all-or-nothing action potentials.
    The mathematical function for this is the Heaviside step: output is 0
    below threshold, 1 at or above threshold. The problem is that this
    function has zero derivative everywhere except at the threshold, where
    it is undefined. Since gradient-based training requires derivatives, we
    replace the Heaviside's derivative (a Dirac delta, useless for training)
    with the derivative of the arctangent function, which is smooth and
    bell-shaped around the threshold.

    The forward pass still produces hard binary spikes (0 or 1). Only the
    backward pass uses the surrogate. This means the network's actual
    dynamics are fully spiking, but gradients can flow through the spike
    function during training.

    CITATION: Neftci, E.O., Mostafa, H., & Zenke, F. (2019). "Surrogate
    gradient learning in spiking neural networks: Bringing the power of
    gradient-based optimization to spiking neural networks." IEEE Signal
    Processing Magazine, 36(6), 51-63. DOI: 10.1109/MSP.2019.2931595

    The specific surrogate used here is:

        forward:  s = Heaviside(v - threshold)
        backward: ds/dv = alpha / (2 * pi * (1 + (alpha * (v - threshold))^2))

    This is the derivative of (1/pi) * arctan(alpha * x) + 0.5, which is a
    smooth approximation to the Heaviside centered at x=0. The alpha
    parameter controls the sharpness: higher alpha makes it more step-like
    (better spike fidelity, smaller gradients far from threshold), lower
    alpha makes it smoother (worse fidelity, larger gradients).
    """

    alpha = 4.0

    @staticmethod
    def forward(ctx, v_mem: Tensor, v_threshold: Tensor) -> Tensor:
        """Emit a binary spike where membrane potential exceeds threshold.

        Parameters
        ----------
        v_mem : Tensor, shape (batch, neurons)
            Current membrane potential.
        v_threshold : Tensor, shape (neurons,)
            Per-neuron spike threshold (learnable, clamped).

        Returns
        -------
        spikes : Tensor, shape (batch, neurons)
            Binary spike tensor (0.0 or 1.0), same dtype as v_mem.
        """
        ctx.save_for_backward(v_mem, v_threshold)
        return (v_mem >= v_threshold).to(v_mem.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute surrogate gradient for membrane potential and threshold.

        The gradient with respect to the threshold is the negative of the
        gradient with respect to the membrane potential, because increasing
        the threshold has the opposite effect of increasing the potential:
        both move the (v - threshold) distance but in opposite directions.

        Parameters
        ----------
        grad_output : Tensor
            Upstream gradient from the loss.

        Returns
        -------
        grad_v_mem : Tensor
            Surrogate gradient with respect to membrane potential.
        grad_v_threshold : Tensor
            Surrogate gradient with respect to threshold (= -grad_v_mem).
        """
        v_mem, v_threshold = ctx.saved_tensors
        x = (v_mem.float() - v_threshold.float())
        surrogate_grad = ATanSurrogate.alpha / (
            2.0 * math.pi * (1.0 + (ATanSurrogate.alpha * x) ** 2)
        )
        grad_v = (grad_output.float() * surrogate_grad).to(v_mem.dtype)
        return grad_v, -grad_v


def spike_fn(v_mem: Tensor, v_threshold: Tensor, alpha: float = 4.0) -> Tensor:
    """Convenience wrapper for ATanSurrogate.apply().

    Parameters
    ----------
    v_mem : Tensor, shape (batch, neurons)
        Current membrane potential.
    v_threshold : Tensor, shape (neurons,)
        Per-neuron spike threshold.
    alpha : float
        Surrogate gradient sharpness. Default 4.0.

    Returns
    -------
    spikes : Tensor, shape (batch, neurons)
        Binary spike tensor (0.0 or 1.0).
    """
    ATanSurrogate.alpha = alpha
    return ATanSurrogate.apply(v_mem, v_threshold)


# =============================================================================
# ASSOCIATIVE LIF NEURON
# =============================================================================

class AssociativeLIF(nn.Module):
    """Leaky Integrate-and-Fire neuron with synaptic filtering and cascade amplification.

    BIOLOGICAL STRUCTURE: Cortical pyramidal cell with first-order synaptic
    current dynamics, absolute refractory period, and lateral excitatory
    connections within a minicolumn cluster.

    BIOLOGICAL FUNCTION: The LIF neuron is the standard computational model
    of a spiking neuron. It captures the essential dynamics of biological
    neurons: subthreshold membrane potential integration, threshold-based
    spike generation, post-spike reset, and refractory silence. The
    "associative" in this class name refers to the cascade amplification
    mechanism, which couples nearby neurons through lateral excitation and
    produces population-level associative dynamics beyond independent firing.

    Gerstner, W., Kistler, W.M., Naud, R., & Paninski, L. (2014).
    "Neuronal Dynamics: From single neurons to networks and models of
    cognition." Cambridge University Press.
    DOI: 10.1017/CBO9781107447615

    COMPUTATIONAL IMPLEMENTATION:

    The neuron dynamics per timestep are:

        1. Synaptic current update:
           i_syn = tau_syn * i_syn + input_current

        2. Membrane potential update (non-refractory neurons only):
           v_mem = tau_mem * v_mem + (1 - tau_mem) * i_syn

        3. Spike generation:
           spike = 1 if v_mem >= threshold, else 0

        4. Cascade amplification (if any neuron spiked):
           i_syn += lateral_excitation_from_neighboring_clusters(spike)

        5. Post-spike reset:
           v_mem = v_mem - spike * threshold  (soft reset)

        6. Refractory counter update:
           If spiked: set refractory counter to refractory_t
           Else: decrement counter by 1 (floor at 0)

    STATE PERSISTENCE (MEM 1):

    When persistent=True, the membrane potential and synaptic current are
    carried between forward() calls via registered buffers. This is how
    Timmy maintains temporal continuity across sequential chunks of input.
    When persistent=False, state is initialized to zero each call (used
    for non-persistent layers like gating circuits).

    The buffers are always registered in state_dict() regardless of the
    persistent flag. When persistent=False, they are stored as (0, 0)
    sentinels that are never read. This ensures that checkpoint loading
    with strict=True never fails due to missing keys.

    INTERFACE BOUNDARY:
        INPUT:  Current tensor of shape (T, batch, neurons)
        OUTPUT: Spike tensor (T, batch, neurons) and membrane trace (T, batch, neurons)
    """

    def __init__(
        self,
        n_neurons: int,
        cfg: NeuronConfig,
        persistent: bool = False,
        tau_mem_override: Optional[float] = None,
    ):
        """Initialize the LIF neuron population.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in this population. In Timmy, this is typically
            d_model (496) for each layer.
        cfg : NeuronConfig
            Neuron configuration with all biophysical and training parameters.
        persistent : bool
            If True, membrane and synaptic state persist between forward()
            calls. Used for layers that must carry temporal state across
            sequential input chunks. If False, state resets each call.
        tau_mem_override : float, optional
            If provided, overrides cfg.tau_mem for this specific population.
            Used by the MemoryCortex to create slow-decaying memory neurons
            with longer integration windows than the default.
        """
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.persistent = persistent

        # -----------------------------------------------------------------
        # v_threshold_raw: (n_neurons,) - learnable per-neuron spike threshold
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Action Potential Threshold
        #
        # PLAIN ENGLISH: The voltage at which the neuron fires. In biological
        # neurons, this is approximately -55mV to -40mV. Here it is in
        # arbitrary units because we do not model absolute voltage levels.
        # Each neuron learns its own threshold during training, allowing the
        # network to develop heterogeneous firing properties across the
        # population.
        #
        # CITATION: Gerstner, W. et al. (2014). "Neuronal Dynamics."
        # Cambridge University Press. DOI: 10.1017/CBO9781107447615
        # -----------------------------------------------------------------
        self.v_threshold_raw = nn.Parameter(
            torch.full((n_neurons,), cfg.v_threshold)
        )

        # -----------------------------------------------------------------
        # beta_mem_raw: scalar - learnable membrane decay (logit-space)
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Membrane Time Constant (inverse, parameterized)
        #
        # PLAIN ENGLISH: Controls how quickly the membrane potential decays
        # toward rest. Stored in logit space so that sigmoid(beta_mem_raw)
        # is always in (0, 1), then clamped to [tau_mem_min, tau_mem_max].
        # A higher value means the membrane retains charge longer (longer
        # integration window). A lower value means faster leak.
        #
        # CITATION: Gerstner, W. et al. (2014). DOI: 10.1017/CBO9781107447615
        # -----------------------------------------------------------------
        tau_mem = tau_mem_override if tau_mem_override is not None else cfg.tau_mem
        self.beta_mem_raw = nn.Parameter(
            torch.tensor(math.log(tau_mem / (1.0 - tau_mem + 1e-6)))
        )

        # -----------------------------------------------------------------
        # beta_syn_raw: scalar - learnable synaptic decay (logit-space)
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Synaptic Time Constant (inverse, parameterized)
        #
        # PLAIN ENGLISH: Controls how quickly the postsynaptic current decays
        # after input arrives. Fast synapses (low tau_syn) respond quickly
        # to each input spike and forget it quickly. Slow synapses (high
        # tau_syn) smooth over multiple input spikes, integrating them.
        #
        # CITATION: Gerstner, W. et al. (2014). DOI: 10.1017/CBO9781107447615
        # -----------------------------------------------------------------
        self.beta_syn_raw = nn.Parameter(
            torch.tensor(math.log(cfg.tau_syn / (1.0 - cfg.tau_syn + 1e-6)))
        )

        # -----------------------------------------------------------------
        # Cascade amplification: minicolumn lateral excitation
        # -----------------------------------------------------------------
        # BIOLOGICAL NAME: Lateral Excitatory Connections (Minicolumn Model)
        #
        # PLAIN ENGLISH: Neurons are grouped into clusters (minicolumn
        # analogs). When neurons in one cluster fire, they send excitatory
        # current to neighboring clusters within cascade_radius. The
        # strength decays linearly with distance: adjacent clusters get
        # the most excitation, clusters at the edge of the radius get the
        # least.
        #
        # In biological cortex, a minicolumn is a vertical column of ~80-100
        # neurons spanning all six cortical layers, sharing common thalamic
        # input and response properties. Lateral excitation between adjacent
        # minicolumns (via horizontal axonal arbors) produces correlated
        # population activity.
        #
        # CITATION: Mountcastle, V.B. (1997). "The columnar organization of
        # the neocortex." Brain, 120(4), 701-722.
        # DOI: 10.1093/brain/120.4.701
        # -----------------------------------------------------------------
        nc = cfg.n_clusters
        self.register_buffer(
            "cluster_ids",
            torch.arange(n_neurons) % nc
        )

        # Build the neighbor excitation weight matrix (nc x nc).
        # neighbor_weights[i, j] > 0 if cluster j is within cascade_radius
        # of cluster i. The weight decays linearly: 1.0 for immediate
        # neighbor, 0.5 at radius/2, 0.0 beyond the radius.
        # The matrix wraps circularly so cluster 0 and cluster (nc-1) are
        # neighbors, which matches the toroidal topology often assumed in
        # cortical sheet models.
        r = cfg.cascade_radius
        idx = torch.arange(nc)
        initial_weights = torch.zeros(nc, nc)
        for offset in range(-r, r + 1):
            if offset != 0:
                initial_weights[idx, (idx + offset) % nc] = (
                    1.0 - abs(offset) / (r + 1)
                )
        self.neighbor_weights = nn.Parameter(initial_weights)

        # Per-cluster gain for cascade amplification.
        self.cluster_gain = nn.Parameter(
            torch.full((nc,), cfg.cascade_gain)
        )

        # -----------------------------------------------------------------
        # Persistent state buffers (MEM 1)
        # -----------------------------------------------------------------
        # Always registered so state_dict() has consistent keys for
        # checkpoint loading. When persistent=False, stored as (0, 0)
        # sentinels that are never read during forward().
        if persistent:
            self.register_buffer("_v_mem_state", torch.zeros(1, n_neurons))
            self.register_buffer("_i_syn_state", torch.zeros(1, n_neurons))
        else:
            self.register_buffer("_v_mem_state", torch.zeros(0, 0))
            self.register_buffer("_i_syn_state", torch.zeros(0, 0))

        # Firing rate tracker (exponential moving average)
        self.register_buffer(
            "_firing_rate_ema",
            torch.full((n_neurons,), cfg.target_spike_rate)
        )
        self.register_buffer(
            "_step_counter",
            torch.tensor(0, dtype=torch.long)
        )

    # --- Clamped property accessors ---

    @property
    def v_threshold(self) -> Tensor:
        """Per-neuron spike threshold, clamped to [v_thresh_min, v_thresh_max]."""
        return self.v_threshold_raw.clamp(
            self.cfg.v_thresh_min, self.cfg.v_thresh_max
        )

    @property
    def beta_mem(self) -> Tensor:
        """Membrane decay factor, clamped to [tau_mem_min, tau_mem_max].

        Stored in logit space (beta_mem_raw), converted via sigmoid and
        then clamped. This double-bounded parameterization prevents the
        membrane from becoming either a perfect integrator (tau=1.0,
        unbounded growth) or an instant-leak wire (tau=0.0, no memory).
        """
        return torch.sigmoid(self.beta_mem_raw).clamp(
            self.cfg.tau_mem_min, self.cfg.tau_mem_max
        )

    @property
    def beta_syn(self) -> Tensor:
        """Synaptic current decay factor in (0, 1), via sigmoid of beta_syn_raw."""
        return torch.sigmoid(self.beta_syn_raw)

    # --- Internal methods ---

    def _cascade_amplify(self, spikes: Tensor) -> Tensor:
        """Compute lateral excitatory current from spiking neurons to neighbors.

        BIOLOGICAL ANALOG: Horizontal axonal arbors within and between
        adjacent cortical minicolumns. When neurons in a minicolumn fire,
        their lateral connections excite neurons in neighboring minicolumns,
        producing correlated population bursting.

        Mountcastle, V.B. (1997). DOI: 10.1093/brain/120.4.701

        The computation:
            1. Aggregate spike rates per cluster (scatter_add over cluster_ids)
            2. Spread activation to neighbors via the weight matrix
            3. Scale by per-cluster gain
            4. Redistribute back to individual neurons via gather

        Parameters
        ----------
        spikes : Tensor, shape (batch, n_neurons)
            Binary spike tensor from the current timestep.

        Returns
        -------
        lateral_current : Tensor, shape (batch, n_neurons)
            Additional synaptic current injected into each neuron from
            the cascade amplification.
        """
        B, D = spikes.shape
        nc = self.cfg.n_clusters
        cid = self.cluster_ids.unsqueeze(0).expand(B, -1)

        # Aggregate per-cluster spike counts, normalized by cluster size
        cluster_fire_rate = torch.zeros(
            B, nc, device=spikes.device, dtype=spikes.dtype
        )
        cluster_fire_rate.scatter_add_(1, cid, spikes)
        cluster_fire_rate = cluster_fire_rate / max(D // nc, 1)

        # Spread to neighbors via sigmoid-gated weight matrix
        W = torch.sigmoid(self.neighbor_weights)
        neighbor_signal = (
            (W.to(cluster_fire_rate.dtype) @ cluster_fire_rate.T).T
            * self.cluster_gain.to(cluster_fire_rate.dtype).unsqueeze(0)
        )

        # Map back from cluster space to neuron space
        return neighbor_signal.gather(1, cid)

    # --- Public interface ---

    def reset_state(self):
        """Reset persistent membrane and synaptic state to zero.

        Called between episodes or during explicit state clearing. Only
        meaningful when persistent=True; no-op otherwise.
        """
        if self.persistent:
            self._v_mem_state.zero_()
            self._i_syn_state.zero_()

    def forward(self, i_input: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the LIF dynamics for T timesteps.

        BIOLOGICAL ANALOG: One theta cycle of cortical processing. The T=8
        timesteps correspond roughly to one oscillatory period during which
        the neuron population receives input, integrates, fires (or not),
        and settles. The temporal structure within this window carries
        information: neurons that fire early in the window respond to
        different features than neurons that fire late.

        Parameters
        ----------
        i_input : Tensor, shape (T, batch, n_neurons)
            Input current for each timestep. This comes from the upstream
            synaptic projection (e.g., the TemporalSpikeEncoder for the
            first layer, or the previous layer's spike output projected
            through a weight matrix for deeper layers).

        Returns
        -------
        spikes : Tensor, shape (T, batch, n_neurons)
            Binary spike output at each timestep.
        v_trace : Tensor, shape (T, batch, n_neurons)
            Membrane potential trace at each timestep. Used by the STDP
            engine for computing eligibility traces and by diagnostics
            for monitoring neuron health.
        """
        T, B, D = i_input.shape
        device, dtype = i_input.device, i_input.dtype
        bm = self.beta_mem
        bs = self.beta_syn
        thresh = self.v_threshold

        # Initialize or restore membrane and synaptic state
        if self.persistent and self._v_mem_state.shape[0] == B:
            v_mem = self._v_mem_state.clone()
            i_syn = self._i_syn_state.clone()
        else:
            v_mem = torch.zeros(B, D, device=device, dtype=dtype)
            i_syn = torch.zeros(B, D, device=device, dtype=dtype)
            if self.persistent:
                self._v_mem_state = torch.zeros(B, D, device=device, dtype=dtype)
                self._i_syn_state = torch.zeros(B, D, device=device, dtype=dtype)

        # Refractory counter (integer, not differentiable)
        refrac = torch.zeros(B, D, device=device, dtype=torch.int32)
        refractory_val = torch.full_like(v_mem, self.cfg.v_reset)
        ref_t = self.cfg.refractory_t
        alpha = self.cfg.surrogate_alpha

        spikes_out = []
        v_trace = []

        for t in range(T):
            # Step 1: Synaptic current update
            # i_syn decays exponentially and accumulates new input
            i_syn = bs * i_syn + i_input[t]

            # Step 2: Membrane potential update
            # Refractory neurons are clamped to reset voltage
            refractory_mask = refrac > 0
            v_new = bm * v_mem + (1.0 - bm) * i_syn
            v_mem = torch.where(refractory_mask, refractory_val, v_new)

            # Step 3: Spike generation (surrogate gradient in backward pass)
            s = spike_fn(v_mem, thresh, alpha)

            # Step 4: Cascade amplification (only if any neuron fired)
            if s.sum() > 0:
                i_syn = i_syn + self._cascade_amplify(s)

            # Step 5: Soft reset (subtract threshold from spiking neurons)
            # Using soft reset (v = v - threshold) rather than hard reset
            # (v = v_reset) preserves the "residual" above threshold, which
            # carries information about input strength.
            v_mem = v_mem - s * thresh.detach()

            # Step 6: Refractory counter update
            refrac = torch.where(
                s.bool(),
                torch.full_like(refrac, ref_t),
                (refrac - 1).clamp(min=0),
            )

            spikes_out.append(s)
            v_trace.append(v_mem)

        # Persist state for next call (detach to avoid graph retention)
        if self.persistent:
            self._v_mem_state = v_mem.detach()
            self._i_syn_state = i_syn.detach()

        # Stack outputs: (T, B, D)
        spike_stack = torch.stack(spikes_out)

        # Update firing rate tracker (no gradient)
        with torch.no_grad():
            self._firing_rate_ema.lerp_(spike_stack.mean(dim=(0, 1)), 0.01)
            self._step_counter += 1

        return spike_stack, torch.stack(v_trace)

    def get_diagnostics(self) -> Dict[str, object]:
        """Return neuron population health diagnostics.

        Returns
        -------
        diagnostics : dict
            mean_firing_rate:    population average firing rate (EMA)
            threshold_mean:      average spike threshold across population
            threshold_std:       spread of thresholds (heterogeneity measure)
            beta_mem:            current membrane decay factor
            beta_syn:            current synaptic decay factor
            steps:               total forward passes since initialization
        """
        return {
            "mean_firing_rate": self._firing_rate_ema.mean().item(),
            "threshold_mean": self.v_threshold.mean().item(),
            "threshold_std": self.v_threshold.std().item(),
            "beta_mem": self.beta_mem.item(),
            "beta_syn": self.beta_syn.item(),
            "steps": self._step_counter.item(),
        }


# =============================================================================
# SMOKE TEST
# =============================================================================

def _smoke_test():
    """Verify the neuron model produces valid spikes and state persistence."""
    print("=" * 60)
    print("Timmy Neuron Module Smoke Test")
    print("=" * 60)

    cfg = NeuronConfig()
    n_neurons = 496
    T = 8
    batch = 2

    # --- Test 1: Basic forward pass ---
    print(f"\n[Test 1] Basic forward pass (non-persistent)")
    lif = AssociativeLIF(n_neurons, cfg, persistent=False)
    total_params = sum(p.numel() for p in lif.parameters())
    print(f"  Parameters: {total_params:,}")

    current = torch.randn(T, batch, n_neurons) * 0.3
    spikes, v_trace = lif(current)
    print(f"  Input:  {tuple(current.shape)}")
    print(f"  Spikes: {tuple(spikes.shape)}")
    print(f"  V_trace: {tuple(v_trace.shape)}")
    assert spikes.shape == (T, batch, n_neurons), "FAIL: Wrong spike shape!"
    assert v_trace.shape == (T, batch, n_neurons), "FAIL: Wrong trace shape!"

    # Spikes should be binary
    unique_vals = torch.unique(spikes)
    assert all(v in [0.0, 1.0] for v in unique_vals.tolist()), "FAIL: Non-binary spikes!"
    rate = spikes.mean().item()
    print(f"  Firing rate: {rate:.4f}")
    print(f"  PASS")

    # --- Test 2: Persistent state ---
    print(f"\n[Test 2] Persistent state across calls")
    lif_p = AssociativeLIF(n_neurons, cfg, persistent=True)

    chunk1 = torch.randn(T, batch, n_neurons) * 0.3
    chunk2 = torch.randn(T, batch, n_neurons) * 0.3

    # Two sequential chunks
    spikes1, _ = lif_p(chunk1)
    v_after_chunk1 = lif_p._v_mem_state.clone()
    spikes2, _ = lif_p(chunk2)
    v_after_chunk2 = lif_p._v_mem_state.clone()

    # State should have changed between chunks
    state_changed = not torch.allclose(v_after_chunk1, v_after_chunk2)
    print(f"  State changed between chunks: {state_changed}")
    assert state_changed, "FAIL: Persistent state did not update!"

    # Reset and verify
    lif_p.reset_state()
    assert lif_p._v_mem_state.sum().item() == 0.0, "FAIL: Reset did not zero state!"
    print(f"  Reset successful")
    print(f"  PASS")

    # --- Test 3: Cascade amplification ---
    print(f"\n[Test 3] Cascade amplification")
    # Feed strong input that guarantees spikes
    strong_input = torch.ones(T, batch, n_neurons) * 0.5
    spikes_strong, _ = lif(strong_input)
    rate_strong = spikes_strong.mean().item()
    print(f"  Strong input firing rate: {rate_strong:.4f}")
    assert rate_strong > 0.01, "FAIL: No spikes even with strong input!"
    print(f"  PASS")

    # --- Test 4: Serialization ---
    print(f"\n[Test 4] Serialization round-trip")
    state = lif_p.state_dict()
    lif_restored = AssociativeLIF(n_neurons, cfg, persistent=True)
    lif_restored.load_state_dict(state)

    test_input = torch.randn(T, 1, n_neurons) * 0.3
    out1, _ = lif_p(test_input)
    lif_p.reset_state()
    lif_restored.reset_state()
    out1b, _ = lif_p(test_input)
    out2, _ = lif_restored(test_input)
    diff = torch.norm(out1b - out2).item()
    print(f"  Output difference after restore: {diff:.12f}")
    assert diff < 1e-5, "FAIL: Serialization corrupted state!"
    print(f"  PASS")

    # --- Test 5: Diagnostics ---
    print(f"\n[Test 5] Diagnostics")
    diag = lif.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")

    print(f"\n{'=' * 60}")
    print(f"ALL SMOKE TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _smoke_test()
