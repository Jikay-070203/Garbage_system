class C2f(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_5.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_8.Conv
  m : __torch__.torch.nn.modules.container.ModuleList
  def forward(self: __torch__.ultralytics.nn.modules.block.C2f,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    m = self.m
    _1 = getattr(m, "1")
    m0 = self.m
    _0 = getattr(m0, "0")
    cv1 = self.cv1
    _2 = (cv1).forward(argument_1, argument_2, )
    _3 = torch.split_with_sizes(_2, [48, 48], 1)
    _4, input, = _3
    _5 = (_0).forward(argument_1, input, )
    _6 = [_4, input, _5, (_1).forward(argument_1, _5, )]
    input0 = torch.cat(_6, 1)
    return (cv2).forward(argument_1, input0, )
class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_11.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_14.Conv
  def forward(self: __torch__.ultralytics.nn.modules.block.Bottleneck,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    input: Tensor) -> Tensor:
    cv2 = self.cv2
    cv1 = self.cv1
    _5 = (cv2).forward(argument_1, (cv1).forward(argument_1, input, ), )
    return torch.add(input, _5)
class SPPF(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_127.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_130.Conv
  m : __torch__.torch.nn.modules.pooling.MaxPool2d
  def forward(self: __torch__.ultralytics.nn.modules.block.SPPF,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _6 = (cv1).forward(argument_1, argument_2, )
    _7 = (m).forward(_6, )
    _8 = (m).forward1(_7, )
    input = torch.cat([_6, _7, _8, (m).forward2(_8, )], 1)
    return (cv2).forward(argument_1, input, )
class DFL(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_278.Conv2d
  def forward(self: __torch__.ultralytics.nn.modules.block.DFL,
    x: Tensor) -> Tensor:
    conv = self.conv
    b = ops.prim.NumToTensor(torch.size(x, 0))
    _9 = int(b)
    _10 = int(b)
    a = ops.prim.NumToTensor(torch.size(x, 2))
    _11 = int(a)
    _12 = torch.transpose(torch.view(x, [_10, 4, 16, int(a)]), 2, 1)
    input = torch.softmax(_12, 1)
    distance = torch.view((conv).forward(input, ), [_9, 4, _11])
    return distance
