import argparse
import datetime

import pyhocon


def read_pyhocon_config(config_path=None):
    import pkg_resources
    default = pkg_resources.resource_filename("fusion", "data/resources/fusion.conf")
    return pyhocon.ConfigFactory().parse_file(default if config_path is None else config_path)


def get_args():
    day = datetime.datetime.utcnow().strftime("%Y%m%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--panel_path", type=str, required=True)
    parser.add_argument("--config_file")
    parser.add_argument("--dt", default=day)
    parser.add_argument("--region", type=str)
    parser.add_argument("--critical_cells", type=str, default="age,gender")
    parser.add_argument("--features", type=str)
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()


class FusionConfig:

    def __init__(self, dt: str = None,
                 region: str = None, conf_file: str = None):

        parsed_args = get_args()
        config_file = parsed_args.config_file if "config_file" in parsed_args else None

        if config_file is None:
            if conf_file is not None:
                config_file = conf_file
            else:
                raise Exception(
                    "This package needs a config file containing"
                    " all input output addresses!"
                )

        self._env_dict = read_pyhocon_config(config_file)["fusion"]

        self.panel_path = parsed_args.panel_path

        self.output_path = parsed_args.output_path

        self.dt = dt if dt else parsed_args.dt

        self.region = region

        self.critical_cells = parsed_args.critical_cells

        self.features = parsed_args.features if parsed_args.features else FUSION_FEATURE_MAP[region]

        self._region_panel_default = REGION_PANEL_MAP[region]

    @property
    def fusible_audiences_api_url(self):
        return self._env_dict["ppc.api.fusible_audiences"]

    @property
    def targets_api_url(self):
        return self._env_dict["ppc.api.targets"]

    @property
    def forecaster_seg_path(self):
        return self._env_dict["ppc.paths.forecaster_seg"]

    @property
    def export_path(self):
        return self._env_dict["ppc.paths.export"]


REGION_PANEL_MAP = {
    "us": "nielsen",
    "uk": "barb",
    "au": "oztam",
    "ru": "tsk"
}

FUSION_FEATURE_MAP = {
    "us": "age,gender,children,education,income",
    "uk": "age,gender,children,marital",
    "au": "age,gender,children,education,income,occupation",
    "ru": "age,gender"
}
