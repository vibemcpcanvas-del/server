"""Phase 2-C: Curriculum learning environments for M8 Jin Hilla training.

JinHillaEasyEnv makes altar cleansing trivially reachable (1 hit required,
frequent altars, slower hazards, longer episodes) so the policy can first
learn the *habit* of moving to the altar and pressing HARVEST before facing
the full-difficulty JinHillaScenarioEnv rules.

Usage in train.py (Phase 2 integration):

    from env.jin_hilla_curriculum_env import JinHillaEasyEnv
    from env.jin_hilla_scenario_env import JinHillaScenarioEnv

    def make_env(episode: int, total_episodes: int):
        # First 40% of training: easy curriculum only.
        # Next 30%: 50/50 mix of easy and full difficulty.
        # Final 30%: full difficulty only.
        progress = episode / max(1, total_episodes)
        if progress < 0.4:
            return JinHillaEasyEnv()
        if progress < 0.7:
            return JinHillaEasyEnv() if random.random() < 0.5 else JinHillaScenarioEnv()
        return JinHillaScenarioEnv()
"""

from __future__ import annotations

from .jin_hilla_scenario_env import JinHillaScenarioEnv


class JinHillaEasyEnv(JinHillaScenarioEnv):
    """Phase 0/curriculum environment: cleansing is easy to discover.

    Differences from JinHillaScenarioEnv:
    - hazard_interval widened -> fewer hits, more time to react.
    - altar_interval unused for spawning (soul-slash driven altar spawning is
      kept, per the M8 altar formula), but max_steps is extended so more
      soul-slash cycles occur per episode, giving more altar opportunities.
    - altar_hits_needed is forced to 1 every soul slash, regardless of
      current green skull count, so a single hit always makes an altar
      spawn-eligible cycle.
    """

    def __init__(self, seed: int = 7) -> None:
        super().__init__(
            seed=seed,
            hazard_interval=20,
            altar_interval=20,
            max_steps=600,
        )

    def _on_soul_slash(self) -> None:
        """Force the easiest possible altar spawning threshold.

        Overrides JinHillaScenarioEnv._on_soul_slash so that, regardless of
        the real green-skull-based formula, this curriculum stage always
        allows altar spawning after a single hit. This intentionally breaks
        real Jin Hilla fidelity in exchange for a much denser learning
        signal for the cleanse behavior; it must only be used for the early
        curriculum stage, never for final M8 pass/fail evaluation.
        """
        state = self._require_state()
        self.hits_since_soul_slash = 0
        if state.souls.green <= 0:
            self.can_spawn_altar = False
            self.altar_hits_needed = None
            return
        self.can_spawn_altar = True
        self.altar_hits_needed = 1
