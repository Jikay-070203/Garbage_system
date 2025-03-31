class Upsample(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.upsampling.Upsample,
    argument_1: Tensor) -> Tensor:
    _0 = torch.upsample_nearest2d(argument_1, None, [2., 2.])
    return _0
