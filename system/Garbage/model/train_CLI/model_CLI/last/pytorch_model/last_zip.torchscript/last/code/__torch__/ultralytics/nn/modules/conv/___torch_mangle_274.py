class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_272.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU
  def forward(self: __torch__.ultralytics.nn.modules.conv.___torch_mangle_274.Conv,
    argument_1: Tensor) -> Tensor:
    act = self.act
    conv = self.conv
    _0 = (act).forward76((conv).forward(argument_1, ), )
    return _0
