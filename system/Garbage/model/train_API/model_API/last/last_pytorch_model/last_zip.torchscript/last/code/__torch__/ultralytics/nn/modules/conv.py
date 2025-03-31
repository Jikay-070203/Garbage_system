class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.Conv2d
  act : __torch__.torch.nn.modules.activation.SiLU
  def forward(self: __torch__.ultralytics.nn.modules.conv.Conv,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    x: Tensor) -> Tensor:
    conv = self.conv
    _0 = (argument_1).forward((conv).forward(x, ), )
    return _0
class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.ultralytics.nn.modules.conv.Concat,
    argument_1: Tensor,
    argument_2: Tensor) -> Tensor:
    input = torch.cat([argument_1, argument_2], 1)
    return input
