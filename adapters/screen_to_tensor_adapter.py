# screen_to_tensor_adapter.py
# 화막에서 읽은 값(타이마, HP%, 스커 건수, 캐릭토 위젬 답)을
# 기족 정책 신안럌이 쒰는 틹서(obs['scalars'], obs['entities'], obs['entity_mask'])로 그낌로 둌첨적게 어덤퇄.
# 정책/학습 코녌려된 것이 아니룜 이 파일마 깔만하동로 실제 화막 데이타로 쟁을 수 있다.

from dataclasses import dataclass
import torch

@dataclass
class ScreenReading:
    timer_seconds_remaining: float
    scythe_seconds_remaining: float
    boss_hp_fraction: float
    green_skulls: int
    red_skulls: int
    altar_present: bool
    altar_lane: int
    player_lane: int
    lane_danger: list
    stunned: bool


def build_observation(reading, device, max_entities=64, entity_dim=12, scalar_dim=32,
                       lane_start=2, altar_entity=9, episode_seconds=20*60, scythe_ref_seconds=182.0):
    entities = torch.zeros(1, max_entities, entity_dim, device=device)
    mask = torch.zeros(1, max_entities, dtype=torch.bool, device=device)
    mask[:, 0] = True
    mask[:, 1] = True
    entities[0, 1, 1] = reading.player_lane * 2.0

    lanes = 7
    for i in range(lanes):
        slot = lane_start + i
        entities[0, slot, 1] = i * 2.0
        entities[0, slot, 9] = 1.0
        entities[0, slot, 10] = 1.0
        entities[0, slot, 11] = 0.0 if reading.lane_danger[i] else 1.0
        mask[0, slot] = True

    if reading.altar_present and reading.altar_lane is not None:
        entities[0, altar_entity, 1] = reading.altar_lane * 2.0
        entities[0, altar_entity, 0] = 2.0
        entities[0, altar_entity, 10] = 1.0
        mask[0, altar_entity] = True

    scalars = torch.zeros(1, scalar_dim, device=device)
    scalars[0, 0] = 1.0 if reading.stunned else 0.0
    scalars[0, 2] = float(any(reading.lane_danger))
    scalars[0, 3] = reading.boss_hp_fraction
    scalars[0, 4] = reading.green_skulls / 5.0
    scalars[0, 5] = reading.red_skulls / 5.0
    scalars[0, 6] = min(1.0, reading.scythe_seconds_remaining / scythe_ref_seconds)
    scalars[0, 10] = float(reading.altar_present)
    scalars[0, 12] = max(0.0, reading.timer_seconds_remaining / episode_seconds)

    return {'scalars': scalars, 'entities': entities, 'entity_mask': mask}
