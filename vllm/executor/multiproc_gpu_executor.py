import asyncio
import os
from functools import partial
<<<<<<< HEAD
from typing import Any, Dict, Optional, Tuple
=======
from typing import Any, List, Optional
>>>>>>> fixie-ai/vllm/main

from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
<<<<<<< HEAD
=======
from vllm.sequence import ExecuteModelRequest, SamplerOutput
>>>>>>> fixie-ai/vllm/main
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, make_async)

logger = init_logger(__name__)


class MultiprocessingGPUExecutor(DistributedGPUExecutor):
    """Python multiprocessing-based multi-GPU executor"""

    def _init_executor(self) -> None:
<<<<<<< HEAD
        assert (
            not self.speculative_config
        ), "Speculative decoding not yet supported for MultiProcGPU backend."

=======
>>>>>>> fixie-ai/vllm/main
        # Create the parallel GPU workers.
        world_size = self.parallel_config.tensor_parallel_size

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = (",".join(
                map(str, range(world_size))))

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

<<<<<<< HEAD
=======
        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

>>>>>>> fixie-ai/vllm/main
        from torch.cuda import device_count
        assert world_size <= device_count(), (
            "please set tensor_parallel_size to less than max local gpu count")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        if world_size == 1:
            self.workers = []
<<<<<<< HEAD
=======
            self.worker_monitor = None
>>>>>>> fixie-ai/vllm/main
        else:
            result_handler = ResultHandler()
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_worker,
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    )) for rank in range(1, world_size)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_worker(
            distributed_init_method=distributed_init_method)
        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

<<<<<<< HEAD
=======
    def _driver_execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(
            execute_model_req=execute_model_req)

>>>>>>> fixie-ai/vllm/main
    def _run_workers(
        self,
        method: str,
        *args,
<<<<<<< HEAD
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
=======
        async_run_remote_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_remote_workers_only: If True the method will be run only
                in the remote workers, not the driver worker. It will also be
                run asynchronously and return a list of futures rather than
                blocking on the results.
        """
>>>>>>> fixie-ai/vllm/main

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

<<<<<<< HEAD
        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Start the driver worker after all the ray workers.
        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*driver_args,
                                                    **driver_kwargs)
=======
        if async_run_remote_workers_only:
            # Just return futures
            return worker_outputs

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)
>>>>>>> fixie-ai/vllm/main

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
<<<<<<< HEAD
        if not self.worker_monitor.is_alive():
            raise RuntimeError("Worker processes are not running")

=======
        if self.worker_monitor is not None and not self.worker_monitor.is_alive(
        ):
            raise RuntimeError("Worker processes are not running")

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

>>>>>>> fixie-ai/vllm/main

class MultiprocessingGPUExecutorAsync(MultiprocessingGPUExecutor,
                                      DistributedGPUExecutorAsync):

<<<<<<< HEAD
    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        driver_executor = make_async(getattr(self.driver_worker, method))

        # Run all the workers asynchronously.
        coros = [driver_executor(*driver_args, **driver_kwargs)] + [
            worker.execute_method_async(method, *args, **kwargs)
            for worker in self.workers
        ]

=======
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        return await self.driver_exec_model(execute_model_req)

    async def _start_worker_execution_loop(self):
        coros = [
            worker.execute_method_async("start_worker_execution_loop")
            for worker in self.workers
        ]
>>>>>>> fixie-ai/vllm/main
        return await asyncio.gather(*coros)
