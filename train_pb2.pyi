from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[DataType]
    FLOAT_16: _ClassVar[DataType]
    FLOAT_32: _ClassVar[DataType]
    FLOAT_64: _ClassVar[DataType]
UNKNOWN: DataType
FLOAT_16: DataType
FLOAT_32: DataType
FLOAT_64: DataType

class RunStepReuest(_message.Message):
    __slots__ = ("epoch", "step", "weights")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    step: int
    weights: _containers.RepeatedCompositeFieldContainer[Ndarray]
    def __init__(self, epoch: _Optional[int] = ..., step: _Optional[int] = ..., weights: _Optional[_Iterable[_Union[Ndarray, _Mapping]]] = ...) -> None: ...

class RunStepResponse(_message.Message):
    __slots__ = ("epoch", "step", "grads", "loss")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    GRADS_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    step: int
    grads: _containers.RepeatedCompositeFieldContainer[TFTensor]
    loss: TFTensor
    def __init__(self, epoch: _Optional[int] = ..., step: _Optional[int] = ..., grads: _Optional[_Iterable[_Union[TFTensor, _Mapping]]] = ..., loss: _Optional[_Union[TFTensor, _Mapping]] = ...) -> None: ...

class TFTensor(_message.Message):
    __slots__ = ("data", "dtype")
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    dtype: DataType
    def __init__(self, data: _Optional[bytes] = ..., dtype: _Optional[_Union[DataType, str]] = ...) -> None: ...

class Ndarray(_message.Message):
    __slots__ = ("data", "dtype", "shape")
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    dtype: DataType
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[bytes] = ..., dtype: _Optional[_Union[DataType, str]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class FinishRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FinishResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HeartBeatRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HeartBeatResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
