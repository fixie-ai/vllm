<<<<<<< HEAD
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
=======
from typing import Any, Dict, Optional, Union
>>>>>>> fixie-ai/vllm/main

import torch
import torch.distributed

<<<<<<< HEAD
from .parallel_state import (get_cpu_world_group,
                             get_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size,
                             get_tp_ca_communicator,
                             get_tp_pynccl_communicator)


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


@contextmanager
def graph_capture():
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    stream = torch.cuda.Stream()
    graph_capture_context = GraphCaptureContext(stream)
    ca_comm = get_tp_ca_communicator()
    maybe_ca_context = nullcontext() if ca_comm is None else ca_comm.capture()
    with torch.cuda.stream(stream), maybe_ca_context:
        # In graph mode, we have to be very careful about the collective
        # operations. The current status is:
        #     allreduce \ Mode   |  Eager  |  Graph  |
        # --------------------------------------------
        # custom allreduce       | enabled | enabled |
        # PyNccl                 | disabled| enabled |
        # torch.distributed      | enabled | disabled|
        #
        # Note that custom allreduce will have a runtime check, if the tensor
        #  size is too large, it will fallback to the next available option.
        # In summary: When using CUDA graph, we use
        # either custom all-reduce kernel or pynccl. When not using CUDA
        # graph, we use either custom all-reduce kernel or PyTorch NCCL.
        # We always prioritize using custom all-reduce kernel but fall back
        # to PyTorch or pynccl if it is disabled or not supported.
        pynccl_comm = get_tp_pynccl_communicator()
        if pynccl_comm is None:
            maybe_pynccl_context = nullcontext()
        else:
            maybe_pynccl_context = pynccl_comm.change_state(
                enable=True, stream=torch.cuda.current_stream())
        with maybe_pynccl_context:
            yield graph_capture_context


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation will be applied in-place on the input tensor if
    disable_custom_all_reduce is set to True. Otherwise, this operation may or
    may not be applied in place depending on whether custom all reduce is
    invoked for a particular tensor, which further depends on the tensor size
    and GPU topology.

    TLDR: always assume this function modifies its input, but use the return
    value as the output.
    """
    ca_comm = get_tp_ca_communicator()

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    if ca_comm is not None:
        out = ca_comm.custom_all_reduce(input_)
        if out is not None:
            return out
    pynccl_comm = get_tp_pynccl_communicator()
    if (pynccl_comm is not None and not pynccl_comm.disabled):
        pynccl_comm.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_,
                                     group=get_tensor_model_parallel_group())
    return input_
=======
from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
>>>>>>> fixie-ai/vllm/main


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


<<<<<<< HEAD

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
    tensor_dict: Dict[Any, Union[torch.Tensor, Any]]
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list = []
    tensor_list = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = "cpu" if value.is_cpu else "cuda"
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size())))
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary.
    `group` is used to broadcast the tensors, while `metadata_group` is used
     to broadcast the metadata of the dict (e.g. dict structure, tensor sizes,
     dtypes).
    """
    # Bypass the function if we are using only 1 GPU.
    if (not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(group=group) == 1):
        return tensor_dict

    group = group or torch.distributed.group.WORLD
    metadata_group = metadata_group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    rank = torch.distributed.get_rank()
    if rank == src:
        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `broadcast_object_list` involves serialization and deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        torch.distributed.broadcast_object_list([metadata_list],
                                                src=src,
                                                group=metadata_group)
        async_handles = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip broadcasting empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                handle = torch.distributed.broadcast(tensor,
                                                     src=src,
                                                     group=metadata_group,
                                                     async_op=True)
            else:
                # use group for GPU tensors
                handle = torch.distributed.broadcast(tensor,
                                                     src=src,
                                                     group=group,
                                                     async_op=True)
            async_handles.append(handle)
        for async_handle in async_handles:
            async_handle.wait()

    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list,
                                                src=src,
                                                group=metadata_group)
        assert recv_metadata_list[0] is not None
        tensor_dict = {}
        async_handles = []
        for key, value in recv_metadata_list[0]:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
=======
def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
>>>>>>> fixie-ai/vllm/main
