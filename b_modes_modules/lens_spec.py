from dataclasses import dataclass

@dataclass(frozen=True)
class LensSpec:
    lens_order: int = 2 # 0 = unlensed
    nside_input: int = 4096
    nside_output: int = 4096
    lmax_cut: int = 1024
    shape_type: str = "proj_tensors"
    min_particles: int = 300

    @property
    def tag(self) -> str:
        return f"lens{self.lens_order}_ns{self.nside_output}_lmax{self.lmax_cut}"
