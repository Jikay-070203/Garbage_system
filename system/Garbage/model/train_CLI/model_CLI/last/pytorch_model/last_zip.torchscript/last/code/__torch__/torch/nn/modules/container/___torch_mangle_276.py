class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.ultralytics.nn.modules.conv.___torch_mangle_271.Conv
  __annotations__["1"] = __torch__.ultralytics.nn.modules.conv.___torch_mangle_274.Conv
  __annotations__["2"] = __torch__.torch.nn.modules.conv.___torch_mangle_275.Conv2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_276.Sequential,
    argument_1: Tensor) -> Tensor:
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _10 = getattr(self, "1")
    act = _10.act
    _0 = getattr(self, "0")
    _3 = (_1).forward((_0).forward(act, argument_1, ), )
    return (_2).forward(_3, )
