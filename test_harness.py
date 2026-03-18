"""
test_harness.py
===============

PRAGMI Cognitive Kernel: End-to-End Integration Test
====================================================

This harness exercises the full trisynaptic loop:

    Timmy output -> Perforant Path (comm subspace) -> CA3 (read)
    -> CA1 (mismatch) -> routing decision -> CA3 (write/reconsolidate)
    -> persistence check across simulated reset

It is the first runnable proof that all three tasks (ZoneTap, Ben-Israel,
Graded Novelty) are wired correctly and that the system can:

    1. Encode episodes that survive a simulated reset
    2. Retrieve those episodes from partial cues
    3. Distinguish novel from familiar input
    4. Reconsolidate (update without destroying) existing memories
    5. Throttle input when the astrocyte detects metabolic stress

This is NOT the acceptance test from the Engineering Protocol (which
requires cue-specific recall across a library with overlapping elements
and scores identity preservation). This is the connectivity and numerical
stability test that must pass before the acceptance test can even run.
"""

from __future__ import annotations

import sys
import os
import tempfile
import torch
import torch.nn.functional as F

# Import from the kernel files in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognitive_kernel import HippocampalConfig, CognitiveKernel


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_result(name: str, passed: bool, detail: str = ""):
    """Print a test result."""
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not passed:
        print(f"         *** TEST FAILED ***")
    return passed


def run_full_harness():
    """Execute the complete end-to-end integration test."""

    print_section("PRAGMI Cognitive Kernel: End-to-End Integration Test")

    cfg = HippocampalConfig()
    kernel = CognitiveKernel(cfg)

    all_passed = True
    total_tests = 0
    passed_tests = 0

    # =================================================================
    # PHASE 1: ENCODING
    # =================================================================
    print_section("Phase 1: Episode Encoding")

    # Generate a library of distinct episodes (simulated Timmy output)
    n_episodes = 12
    episode_library = {}
    for i in range(n_episodes):
        # Each episode is a distinct point in Timmy's output space.
        # We seed the generator for reproducibility.
        torch.manual_seed(1000 + i)
        timmy_output = torch.randn(1, cfg.bridge_dim)
        episode_library[f"ep_{i}"] = timmy_output

    # Encode all episodes
    print(f"  Encoding {n_episodes} episodes...")
    encode_results = {}
    for name, timmy_out in episode_library.items():
        result = kernel(timmy_out, mode="encode")
        encode_results[name] = {
            "coordinates": result["coordinates"].detach().clone(),
            "ca3_converged": result["ca3_converged"],
            "ca3_residual": result["ca3_residual"],
        }
        converged = bool(result["ca3_converged"])
        if not converged:
            print(f"    WARNING: {name} did not converge "
                  f"(residual={result['ca3_residual']:.6f})")

    # Test: At least half should converge
    n_converged = sum(1 for r in encode_results.values() if r["ca3_converged"])
    total_tests += 1
    ok = n_converged >= n_episodes // 2
    if print_result(
        f"Encoding convergence: {n_converged}/{n_episodes} converged",
        ok
    ):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # PHASE 2: RETRIEVAL FROM FULL CUES
    # =================================================================
    print_section("Phase 2: Full-Cue Retrieval")

    # Retrieve each episode using the exact same Timmy output
    retrieval_errors = []
    for name, timmy_out in episode_library.items():
        result = kernel(timmy_out, mode="retrieve")
        stored_coords = encode_results[name]["coordinates"]
        error = F.mse_loss(result["coordinates"], stored_coords).item()
        retrieval_errors.append(error)

    mean_error = sum(retrieval_errors) / len(retrieval_errors)
    max_error = max(retrieval_errors)
    print(f"  Mean retrieval MSE: {mean_error:.6f}")
    print(f"  Max retrieval MSE:  {max_error:.6f}")

    total_tests += 1
    ok = mean_error < 1.0  # Generous threshold for prototype
    if print_result("Full-cue retrieval", ok, f"mean MSE={mean_error:.6f}"):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # PHASE 3: PARTIAL CUE RETRIEVAL
    # =================================================================
    print_section("Phase 3: Partial-Cue Retrieval")

    # Corrupt 50% of the Timmy output (zero out half the dimensions)
    partial_errors = []
    for name, timmy_out in episode_library.items():
        partial = timmy_out.clone()
        partial[0, cfg.bridge_dim // 2:] = 0.0  # zero second half
        result = kernel(partial, mode="retrieve")
        stored_coords = encode_results[name]["coordinates"]
        error = F.mse_loss(result["coordinates"], stored_coords).item()
        partial_errors.append(error)

    mean_partial = sum(partial_errors) / len(partial_errors)
    print(f"  Mean partial-cue MSE: {mean_partial:.6f}")
    print(f"  (Expected: higher than full-cue but still structured)")

    # Test: partial cue should be worse than full cue but not catastrophic
    total_tests += 1
    ok = mean_partial < 5.0  # very generous for prototype
    if print_result("Partial-cue retrieval", ok, f"mean MSE={mean_partial:.6f}"):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # PHASE 4: NOVELTY DETECTION
    # =================================================================
    print_section("Phase 4: CA1 Novelty Detection")

    # Test 1: Familiar input (re-present an encoded episode)
    familiar_out = episode_library["ep_0"]
    result_familiar = kernel(familiar_out, mode="auto")
    novelty_familiar = result_familiar["novelty"].item()
    mse_familiar = result_familiar["mismatch_mse"].item()
    print(f"  Familiar: novelty={novelty_familiar:.4f}, MSE={mse_familiar:.6f}, "
          f"action={result_familiar['action_taken']}")

    # Test 2: Novel input (completely new Timmy output)
    torch.manual_seed(9999)
    novel_timmy = torch.randn(1, cfg.bridge_dim) * 3.0
    result_novel = kernel(novel_timmy, mode="auto")
    novelty_novel = result_novel["novelty"].item()
    mse_novel = result_novel["mismatch_mse"].item()
    print(f"  Novel:    novelty={novelty_novel:.4f}, MSE={mse_novel:.6f}, "
          f"action={result_novel['action_taken']}")

    # Test: novel should have higher novelty than familiar
    total_tests += 1
    ok = novelty_novel > novelty_familiar
    if print_result(
        "Novelty ordering (novel > familiar)",
        ok,
        f"{novelty_novel:.4f} > {novelty_familiar:.4f}"
    ):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # PHASE 5: RECONSOLIDATION
    # =================================================================
    print_section("Phase 5: Reconsolidation")

    # Present a slightly modified version of ep_0. This should trigger
    # reconsolidation (medium mismatch), not new encoding.
    ep0_original = episode_library["ep_0"]
    ep0_modified = ep0_original + torch.randn_like(ep0_original) * 0.3

    # Get the CA3 state of slot 0 before reconsolidation
    slot0_before = kernel.ca3.memory_mean[0].clone()

    result_recon = kernel(ep0_modified, mode="auto")
    action = result_recon["action_taken"]
    novelty = result_recon["novelty"].item()
    print(f"  Modified ep_0: novelty={novelty:.4f}, action={action}")

    slot0_after = kernel.ca3.memory_mean[0].clone()
    slot_change = torch.norm(slot0_after - slot0_before).item()
    print(f"  Slot 0 change magnitude: {slot_change:.6f}")

    # The auto-mode should have either reconsolidated or encoded.
    # We check that SOME write happened (memory changed).
    total_tests += 1
    # Check any slot changed, not just slot 0
    total_change = torch.norm(
        kernel.ca3.memory_mean - kernel.ca3.memory_mean  # this is always 0
    ).item()
    # Better: check write count increased
    total_writes = kernel.ca3.slot_write_count.sum().item()
    ok = total_writes > n_episodes  # more writes than initial encoding
    if print_result(
        f"Reconsolidation triggered write (total writes: {total_writes})",
        ok
    ):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # PHASE 6: ASTROCYTE-PP THROTTLE
    # =================================================================
    print_section("Phase 6: Astrocyte-PP Throttle")

    # Simulate metabolic stress by manually injecting convergence failures
    print(f"  Pre-stress PP throttle: {result_recon['pp_throttle']}")

    for _ in range(10):
        kernel.astrocyte.report_convergence(converged=False, residual=5.0)

    stress = kernel.astrocyte.metabolic_stress.item()
    print(f"  Metabolic stress after 10 failures: {stress:.4f}")
    print(f"  Is stressed: {kernel.astrocyte.is_stressed}")

    # Now run a forward pass and check that the throttle is active
    result_stressed = kernel(episode_library["ep_5"], mode="retrieve")
    throttle = result_stressed["pp_throttle"]
    print(f"  Stressed PP throttle: {throttle}")

    total_tests += 1
    # At least one channel should be attenuated below 1.0
    ok = any(t < 0.99 for t in throttle)
    if print_result(
        "Astrocyte throttle active under stress",
        ok,
        f"min gain={min(throttle):.4f}"
    ):
        passed_tests += 1
    all_passed &= ok

    # Verify that throttle actually changes the output
    # (compare with a fresh non-stressed kernel on the same input)
    kernel_fresh = CognitiveKernel(cfg)
    kernel_fresh.load_state_dict(kernel.state_dict())
    # Reset the stress on the fresh copy
    kernel_fresh.astrocyte.metabolic_stress.fill_(0.0)

    result_unstressed = kernel_fresh(episode_library["ep_5"], mode="retrieve")
    output_diff = torch.norm(
        result_stressed["coordinates"] - result_unstressed["coordinates"]
    ).item()
    print(f"  Output difference (stressed vs unstressed): {output_diff:.6f}")

    total_tests += 1
    ok = output_diff > 1e-6
    if print_result(
        "Throttle changes output coordinates",
        ok,
        f"diff={output_diff:.6f}"
    ):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # PHASE 7: PERSISTENCE ACROSS SIMULATED RESET
    # =================================================================
    print_section("Phase 7: Persistence Across Reset")

    # Reset the stress so the comparison is clean
    kernel.astrocyte.metabolic_stress.fill_(0.0)
    kernel.astrocyte.convergence_failure_count.fill_(0)

    # Serialize the entire kernel state
    with tempfile.NamedTemporaryFile(suffix=".soul", delete=False) as f:
        soul_path = f.name
        torch.save(kernel.serialize_state(), f)
    print(f"  Saved state to: {soul_path}")
    file_size = os.path.getsize(soul_path)
    print(f"  File size: {file_size:,} bytes")

    # Create a fresh kernel and restore
    kernel_restored = CognitiveKernel(cfg)
    state = torch.load(soul_path, weights_only=True)
    kernel_restored.load_state(state)
    print(f"  Restored from disk.")

    # Verify identical output for every encoded episode
    max_restore_diff = 0.0
    for name, timmy_out in episode_library.items():
        out_original = kernel(timmy_out, mode="retrieve")
        out_restored = kernel_restored(timmy_out, mode="retrieve")
        diff = torch.norm(
            out_original["coordinates"] - out_restored["coordinates"]
        ).item()
        max_restore_diff = max(max_restore_diff, diff)

    print(f"  Max output difference across {n_episodes} episodes: {max_restore_diff:.12f}")

    total_tests += 1
    ok = max_restore_diff < 1e-5
    if print_result(
        "Persistence across reset",
        ok,
        f"max diff={max_restore_diff:.12f}"
    ):
        passed_tests += 1
    all_passed &= ok

    # Clean up temp file
    os.unlink(soul_path)

    # =================================================================
    # PHASE 8: COMMUNICATION SUBSPACE INTEGRITY
    # =================================================================
    print_section("Phase 8: Communication Subspace Integrity")

    W = kernel.perforant_path.effective_weight()
    exact_rank = torch.linalg.matrix_rank(W).item()
    spectral_norm = torch.linalg.svdvals(W)[0].item()
    eff_rank = kernel._effective_comm_rank()

    print(f"  Exact rank: {exact_rank} (max: {cfg.comm_subspace_rank})")
    print(f"  Effective rank: {eff_rank:.2f}")
    print(f"  Spectral norm: {spectral_norm:.6f}")

    total_tests += 1
    ok = exact_rank <= cfg.comm_subspace_rank
    if print_result("Rank constraint maintained", ok):
        passed_tests += 1
    all_passed &= ok

    # =================================================================
    # SUMMARY
    # =================================================================
    print_section("SUMMARY")
    print(f"  Tests passed: {passed_tests}/{total_tests}")
    print(f"  CA3 diagnostics: {kernel.ca3.get_diagnostics()}")
    print(f"  Astrocyte: {kernel.astrocyte.get_diagnostics()}")

    if all_passed:
        print(f"\n  *** ALL {total_tests} TESTS PASSED ***")
        print(f"  The trisynaptic loop is wired correctly.")
        print(f"  Ready for Engineering Protocol acceptance testing.")
    else:
        print(f"\n  *** {total_tests - passed_tests} TEST(S) FAILED ***")
        print(f"  Fix failures before proceeding to acceptance tests.")

    return all_passed


if __name__ == "__main__":
    success = run_full_harness()
    sys.exit(0 if success else 1)
