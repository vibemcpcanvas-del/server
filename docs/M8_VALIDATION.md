# M8 validation

M8 separates the confirmed combat rules from the learning interface:

- `env/jin_hilla_environment_core.py` contains the skull, altar, and scythe state machine.
- `env/jin_hilla_training_env.py` exposes the state machine through `reset()` and `step()` for a reinforcement-learning loop.
- `tests/test_jin_hilla_m8.py` validates the rules without third-party test packages.

## Run the checks

From the repository root, run:

```bash
python -m unittest tests.test_jin_hilla_m8 -v
```

The suite checks these accepted M8 conditions:

1. A web/thread hit changes one green skull to one red skull.
2. The episode terminates when red skulls outnumber green skulls.
3. Harvesting only works near the altar and moves red skulls back to green.
4. A scythe sequence enters `WARNING` before `SPREAD_ART`.
5. The training interface exposes a fixed nine-value observation and respects movement and episode bounds.

## Use from a training notebook

```python
from env.jin_hilla_training_env import Action, JinHillaTrainingEnv

env = JinHillaTrainingEnv()
observation, info = env.reset()

while True:
    action = Action.STAY  # Replace with the policy output.
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

`apply_web_hit()` and `spawn_altar(lane)` are explicit scenario events. A live-screen adapter should call them only after it has detected the corresponding in-game event; it must not infer timing values that have not been validated from gameplay data.
