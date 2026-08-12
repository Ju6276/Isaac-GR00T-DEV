"""GR00T N1.7 modality configuration for HumanoidArena V3.1.

The slices in this file are part of the benchmark protocol. Keep them synchronized
with HumanoidArena's UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


STATE_KEYS = ["root_rot6d", "joint_pos", "joint_vel"]
ACTION_KEYS = ["root_xy_delta", "root_z", "root_rot6d", "joint_pos", "hand_binary"]


def _absolute_non_eef() -> ActionConfig:
    # The 40D label is already a canonical reference-motion target. It must not
    # be converted to a delta from observation.state by GR00T's processor.
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


humanoidarena_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        # Use the requested GR00T supervision horizon. Deployment preserves
        # HumanoidArena's per-step canonical 40D action protocol.
        delta_indices=list(range(30)),
        modality_keys=ACTION_KEYS,
        action_configs=[_absolute_non_eef() for _ in ACTION_KEYS],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    humanoidarena_config,
    embodiment_tag=EmbodimentTag.HUMANOIDARENA_G1,
)
