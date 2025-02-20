import copy
import json
from dataclasses import dataclass
from virtualhome.simulation.unity_simulator import comm_unity


@dataclass
class VhTaskGeneratorConfig:
    ENV_SEED: int = 10
    TASKPOOL_JSON_FILEPATH: str = "config/generator/init_pool.json"
    CLASSINFO_JSON_FILEPATH: str = "config/generator/class_info.json"

    UNITY_COMMUNICATION_FILEPATH: str = "F:\\Program\\windows_exec.v2.2.4\\VirtualHome.exe"
    UNITY_COMMUNICATION_PORT: str = 8080


class VhTaskGenerator:
    def __init__(self, cfg: VhTaskGeneratorConfig):
        self.cfg = cfg

        with open(cfg.TASKPOOL_JSON_FILEPATH) as f:
            self.taskpool = json.load(f)

        with open(cfg.CLASSINFO_JSON_FILEPATH) as f:
            self.classinfo = json.load(f)

        self.comm = comm_unity.UnityCommunication(
            port=cfg.UNITY_COMMUNICATION_PORT,
            file_name=cfg.UNITY_COMMUNICATION_FILEPATH
        )



