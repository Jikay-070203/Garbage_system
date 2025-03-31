class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.ultralytics.nn.modules.conv.___torch_mangle_255.Conv
  __annotations__["1"] = __torch__.ultralytics.nn.modules.conv.___torch_mangle_258.Conv
  __annotations__["2"] = __torch__.torch.nn.modules.conv.___torch_mangle_259.Conv2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_260.Sequential,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _3 = (_0).forward(argument_1, argument_2, )
    _4 = (_2).forward((_1).forward(argument_1, _3, ), )
    return _4
