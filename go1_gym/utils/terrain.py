# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice

def roughness_tier_columns(num_cols, num_tiers, weights=None):
    """Contiguous column bands, one per roughness tier, tier 0 first.

    A tier is a set of columns rather than a per-tile flag precisely so that it
    is addressable at runtime: moving an environment between tiers is a tensor
    write, whereas regenerating a tile is impossible once the trimesh has been
    handed to the simulator.

    ``weights`` gives each tier's share of the columns.  It exists because the
    flat tier is not just "one of the tiers": it is the ground the R5 nominal
    twins have to stay on for a whole episode of walking, so it needs to be
    wide enough that they cannot walk out of it.  Equal shares by default.
    """
    if num_tiers <= 0 or num_cols < num_tiers:
        raise ValueError("terrain needs at least one column per roughness tier")
    if weights is None:
        weights = [1.0] * num_tiers
    if len(weights) != num_tiers:
        raise ValueError(f"{len(weights)} tier weights for {num_tiers} tiers")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("roughness tier weights must be finite and nonnegative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("roughness tier weights must sum to something positive")
    bands, start, accumulated = [], 0, 0.0
    for tier, weight in enumerate(weights):
        accumulated += weight
        end = num_cols if tier == num_tiers - 1 else int(round(num_cols * accumulated / total))
        # Every tier needs at least one column, and the last one takes the
        # rounding remainder so no column is left unassigned.
        end = max(end, start + 1)
        end = min(end, num_cols - (num_tiers - 1 - tier))
        bands.append(list(range(start, end)))
        start = end
    if sum(len(b) for b in bands) != num_cols:
        raise ValueError(f"tier bands cover {sum(len(b) for b in bands)} of {num_cols} columns")
    return bands


def tier_of_each_column(bands):
    """Inverse of :func:`roughness_tier_columns`: column index -> tier."""
    lookup = {}
    for tier, band in enumerate(bands):
        for col in band:
            lookup[col] = tier
    return [lookup[c] for c in range(len(lookup))]


class Terrain:
    def __init__(self, cfg, num_robots, eval_cfg=None, num_eval_robots=0) -> None:

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.initialize_terrains()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def load_cfgs(self):
        self._load_cfg(self.cfg)
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)
        self.cfg.x_offset = 0
        self.cfg.rows_offset = 0
        if self.eval_cfg is None:
            return self.cfg.row_indices, self.cfg.col_indices, [], []
        else:
            self._load_cfg(self.eval_cfg)
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows, self.cfg.tot_rows + self.eval_cfg.tot_rows)
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)
            self.eval_cfg.x_offset = self.cfg.tot_rows
            self.eval_cfg.rows_offset = self.cfg.num_rows
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices

    def _load_cfg(self, cfg):
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

    def initialize_terrains(self):
        self._initialize_terrain(self.cfg)
        if self.eval_cfg is not None:
            self._initialize_terrain(self.eval_cfg)

    def _initialize_terrain(self, cfg):
        if getattr(cfg, "roughness_tiers", None):
            self.tiered_roughness_terrain(cfg)
        elif cfg.curriculum:
            self.curriculum(cfg)
        elif cfg.selected:
            self.selected_terrain(cfg)
        else:
            self.randomized_terrain(cfg)

    def tiered_roughness_terrain(self, cfg):
        """Mild rough ground only, in fixed roughness tiers laid out by column.

        Deliberately not one of the branches in ``make_terrain``: this is a
        robustness randomisation, not a terrain-generalisation task.  There are
        no slopes, no stairs, no discrete obstacles and no stepping stones --
        just band-limited height noise whose amplitude is the tier.

        Tier 0 is exactly flat and is where the R5 nominal twins stand.  A twin
        on uneven ground would make the response every other environment is
        being aligned to a moving target, which is the one thing the whole
        consistency objective cannot tolerate.
        """
        tiers = list(cfg.roughness_tiers)
        cfg.roughness_tier_columns = roughness_tier_columns(
            cfg.num_cols, len(tiers), getattr(cfg, "roughness_tier_weights", None)
        )
        column_tier = tier_of_each_column(cfg.roughness_tier_columns)
        for k in range(cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            amplitude = float(tiers[column_tier[j]])
            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)
            if amplitude > 0.0:
                terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude,
                                                    max_height=amplitude, step=0.005,
                                                    downsampled_scale=0.2)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def randomized_terrain(self, cfg):
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def curriculum(self, cfg):
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                choice = j / cfg.num_cols + 0.001

                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                self.add_terrain_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            eval(terrain_type)(terrain, **cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def make_terrain(self, cfg, choice, difficulty, proportions):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=cfg.width_per_env_pixels,
                                           length=cfg.width_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        if choice < proportions[0]:
            if choice < proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        elif choice < proportions[3]:
            if choice < proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            pass
        elif choice < proportions[7]:
            pass
        elif choice < proportions[8]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,
                                                 max_height=cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        x1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset
        x2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset
        y1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
