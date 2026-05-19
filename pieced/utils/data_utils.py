# data_utils.py
import torch
import torch.nn.functional as F
from typing import Callable, List

# ====== 공통 Skeleton 변환 클래스들 ======
class _SkelBase:
    @staticmethod
    def _check_x(x: torch.Tensor):
        assert isinstance(x, torch.Tensor) and x.dim() == 4, "x must be (C,T,V,M)"
        C, _, _, _ = x.shape
        assert C == 3, "C must be 3 (x,y,z)"

class SkeletonResizeTime(_SkelBase):
    """(C,T,V,M) → T를 target_len으로 선형보간"""
    def __init__(self, target_len: int = 64):
        self.target_len = int(target_len)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._check_x(x)
        C, T, V, M = x.shape
        if T == self.target_len:
            return x
        x_vmct = x.permute(2, 3, 0, 1).reshape(V * M, C, T)
        x_vmct = F.interpolate(x_vmct, size=self.target_len, mode="linear", align_corners=False)
        x = x_vmct.reshape(V, M, C, self.target_len).permute(2, 3, 0, 1).contiguous()
        return x

class SkeletonTemporalCrop(_SkelBase):
    """시간축 반사패딩 후 동일 길이 랜덤 크롭(길이 보존)."""
    def __init__(self, padding_ratio: int = 6):
        self.padding_ratio = int(padding_ratio)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._check_x(x)
        C, T, V, M = x.shape
        if self.padding_ratio <= 0:
            return x
        pad = T // self.padding_ratio
        if pad <= 0:
            return x
        left = x[:, :pad].flip(dims=[1])
        right = x[:, -pad:].flip(dims=[1])
        x_pad = torch.cat([left, x, right], dim=1)  # (C, T+2*pad, V, M)
        start = int(torch.randint(low=0, high=2 * pad + 1, size=(1,)).item())
        return x_pad[:, start:start + T]

class SkeletonShear(_SkelBase):
    """C=3 채널축(좌표) 3×3 shear 변환."""
    def __init__(self, amplitude: float = 0.5):
        self.amplitude = float(amplitude)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._check_x(x)
        if self.amplitude <= 0:
            return x
        C, T, V, M = x.shape
        # 비대각 원소를 U[-a,a]에서 샘플
        s1 = [torch.empty(1, device=x.device).uniform_(-self.amplitude, self.amplitude).item() for _ in range(3)]
        s2 = [torch.empty(1, device=x.device).uniform_(-self.amplitude, self.amplitude).item() for _ in range(3)]
        R = torch.tensor(
            [[1.0,   s1[0], s2[0]],
             [s1[1], 1.0,   s2[1]],
             [s1[2], s2[2], 1.0]],
            dtype=x.dtype, device=x.device
        )
        x_flat = x.view(3, -1)          # (3, T*V*M)
        y_flat = R @ x_flat             # (3, T*V*M)
        return y_flat.view(3, T, V, M)

# ====== 공통 파이프라인 빌더 ======
def build_eval_pipeline(target_len: int = 64) -> Callable[[torch.Tensor], torch.Tensor]:
    """검증/온라인평가: 길이 보정만."""
    resize = SkeletonResizeTime(target_len)
    return lambda x: resize(x)

def build_shear_temporalcrop_pipeline(
    target_len: int = 64,
    shear_amplitude: float = 0.5,
    temporal_padding_ratio: int = 6,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """학습용 단일 뷰: Resize → TemporalCrop → Shear"""
    resize = SkeletonResizeTime(target_len)
    crop = SkeletonTemporalCrop(temporal_padding_ratio)
    shear = SkeletonShear(shear_amplitude)

    def _t(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != target_len:
            x = resize(x)
        x = crop(x)
        x = shear(x)
        return x
    return _t

class MulticropSkeletonTransform:
    """BYOL 등 self-supervised에서 쓰는 N-view 생성기 (각 뷰에 동일 규칙 적용)."""
    def __init__(
        self,
        target_len: int = 64,
        num_crops: int = 2,
        shear_amplitude: float = 0.5,
        temporal_padding_ratio: int = 6,
    ):
        builder = build_shear_temporalcrop_pipeline(
            target_len=target_len,
            shear_amplitude=shear_amplitude,
            temporal_padding_ratio=temporal_padding_ratio,
        )
        self.transforms: List[Callable[[torch.Tensor], torch.Tensor]] = [builder for _ in range(num_crops)]

    def __call__(self, x: torch.Tensor):
        return [t(x.clone()) for t in self.transforms]

# (분류/프리트레인 공통) 단일 뷰용 얇은 wrapper
class SkeletonTransform:
    def __init__(self, target_len: int = 64):
        self.f = build_eval_pipeline(target_len)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
