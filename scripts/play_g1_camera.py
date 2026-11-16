"""Play the G1 depth student with a live panel showing its chest depth camera (what the policy sees).

Run from C:\\Users\\ADMIN\\Isaacsim\\IsaacLab:

    .\\env_isaaclab\\Scripts\\python.exe ..\\play_g1_camera.py --num_envs 1

The Isaac Sim window opens with the robot in the viewport and a "Streaming View" panel on the right
with the depth image of robot 0, colored by distance (0.1 m to 3 m, the range the policy uses).

Everything after the script name is passed to ``isaaclab play``, for example:
    --num_envs 4                  more robots (the panel still shows robot 0)
    --real-time                   run at real-time speed
    --viz newton_gl               lighter Newton window instead of Isaac Sim
    physics=isaacsim_physx        PhysX instead of Newton
    --checkpoint <path>           a specific model_*.pt instead of the newest one
"""

import sys

from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import DEPTH_MAX_RANGE, G1RoughDepthStudentEnvCfg

TASK = "Isaac-Velocity-Rough-G1-Depth-Student"


@configclass
class G1DepthStudentCameraViewEnvCfg(G1RoughDepthStudentEnvCfg):
    """The student environment with the chest depth camera shown in the visualizer."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_visualizers.kit import KitBackendCfg

        self.sim.visualizer_cfgs = VisualizerCfg(
            backend=KitBackendCfg(dock_position="RIGHT"),
            streaming_view=True,
            streaming_sensor_prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
            streaming_gt_types=("depth",),
            streaming_envs=[0],
            streaming_depth_min=0.1,
            streaming_depth_max=DEPTH_MAX_RANGE,
        )


if __name__ == "__main__":
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  (registers the tasks)

    from isaaclab_rl.entrypoints.backends import play_rsl_rl

    # Same task name, so "--checkpoint latest" still finds the student's training runs.
    gym.registry[TASK].kwargs["env_cfg_entry_point"] = "play_g1_camera:G1DepthStudentCameraViewEnvCfg"
    defaults = ["--task", TASK, "--checkpoint", "latest", "--num_envs", "1"]
    play_rsl_rl.run(defaults + sys.argv[1:])
