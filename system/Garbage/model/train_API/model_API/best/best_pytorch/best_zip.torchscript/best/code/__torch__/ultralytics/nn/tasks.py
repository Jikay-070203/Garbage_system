class DetectionModel(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  model : __torch__.torch.nn.modules.container.___torch_mangle_279.Sequential
  def forward(self: __torch__.ultralytics.nn.tasks.DetectionModel,
    x: Tensor) -> Tensor:
    model = self.model
    _22 = getattr(model, "22")
    model0 = self.model
    _21 = getattr(model0, "21")
    model1 = self.model
    _20 = getattr(model1, "20")
    model2 = self.model
    _19 = getattr(model2, "19")
    model3 = self.model
    _18 = getattr(model3, "18")
    model4 = self.model
    _17 = getattr(model4, "17")
    model5 = self.model
    _16 = getattr(model5, "16")
    model6 = self.model
    _15 = getattr(model6, "15")
    model7 = self.model
    _14 = getattr(model7, "14")
    model8 = self.model
    _13 = getattr(model8, "13")
    model9 = self.model
    _12 = getattr(model9, "12")
    model10 = self.model
    _11 = getattr(model10, "11")
    model11 = self.model
    _10 = getattr(model11, "10")
    model12 = self.model
    _9 = getattr(model12, "9")
    model13 = self.model
    _8 = getattr(model13, "8")
    model14 = self.model
    _7 = getattr(model14, "7")
    model15 = self.model
    _6 = getattr(model15, "6")
    model16 = self.model
    _5 = getattr(model16, "5")
    model17 = self.model
    _4 = getattr(model17, "4")
    model18 = self.model
    _3 = getattr(model18, "3")
    model19 = self.model
    _2 = getattr(model19, "2")
    model20 = self.model
    _1 = getattr(model20, "1")
    model21 = self.model
    _220 = getattr(model21, "22")
    cv3 = _220.cv3
    _23 = getattr(cv3, "2")
    _110 = getattr(_23, "1")
    act = _110.act
    model22 = self.model
    _0 = getattr(model22, "0")
    _24 = (_1).forward(act, (_0).forward(act, x, ), )
    _25 = (_3).forward(act, (_2).forward(act, _24, ), )
    _26 = (_4).forward(act, _25, )
    _27 = (_6).forward(act, (_5).forward(act, _26, ), )
    _28 = (_8).forward(act, (_7).forward(act, _27, ), )
    _29 = (_9).forward(act, _28, )
    _30 = (_11).forward((_10).forward(_29, ), _27, )
    _31 = (_12).forward(act, _30, )
    _32 = (_14).forward((_13).forward(_31, ), _26, )
    _33 = (_15).forward(act, _32, )
    _34 = (_17).forward((_16).forward(act, _33, ), _31, )
    _35 = (_18).forward(act, _34, )
    _36 = (_20).forward((_19).forward(act, _35, ), _29, )
    _37 = (_22).forward(_33, _35, (_21).forward(act, _36, ), )
    return _37
