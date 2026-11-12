import pytest
from VLM.quantization.awq_config import AWQConfig


@pytest.mark.parametrize('value', [True, False, '0.5', None, float('nan'), float('inf')])
def test_clip_shrink_requires_finite_number(value):
    with pytest.raises(ValueError):
        AWQConfig(max_clip_shrink=value).validate()
