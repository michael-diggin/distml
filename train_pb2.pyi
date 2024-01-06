from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RunStepReuest(_message.Message):
    __slots__ = ("epoch", "step", "data")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    step: int
    data: bytes
    def __init__(self, epoch: _Optional[int] = ..., step: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class RunStepResponse(_message.Message):
    __slots__ = ("epoch", "step", "grads", "loss")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    GRADS_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    step: int
    grads: bytes
    loss: bytes
    def __init__(self, epoch: _Optional[int] = ..., step: _Optional[int] = ..., grads: _Optional[bytes] = ..., loss: _Optional[bytes] = ...) -> None: ...
