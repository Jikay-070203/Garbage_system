class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_170.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_173.Conv
  def forward(self: __torch__.ultralytics.nn.modules.block.___torch_mangle_174.Bottleneck,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    cv1 = self.cv1
    _0 = (cv1).forward(argument_1, argument_2, )
    return (cv2).forward(argument_1, _0, )
