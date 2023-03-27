import concurrent.futures
import asyncio
import sys
import time
import dataclasses
import signal
from types import FrameType
from typing import Any, Callable


@dataclasses.dataclass()
class TaskSpec:
    function: Callable[..., Any]


def test_task() -> None:
    print("Starting task")
    t0 = time.perf_counter()
    try:
        time.sleep(10)
    finally:
        t1 = time.perf_counter()
        print(f"Task finished in {t1 - t0:.2f} seconds")
        print(sys.exc_info())


num_sigints = 0


def handle_sigint_in_executor(_signal: int, _frame: FrameType | None) -> None:
    global num_sigints

    num_sigints += 1
    if num_sigints > 1:
        print(f"Received more than one sigint, exiting immediately")
        raise KeyboardInterrupt


def initialize_executor() -> None:
    # Ignore SIGINT in the executors. We want to have control over the
    # shutdown, rather than have the workers exit at unexpected times.
    signal.signal(signal.SIGINT, handle_sigint_in_executor)


class TaskRunner:
    def __init__(self) -> None:
        # Queue for pending tasks. To be replaced with an external task queue
        self.queue = asyncio.Queue[TaskSpec]()
        # A process pool that we run tasks in
        self.executor = concurrent.futures.ProcessPoolExecutor(
            initializer=initialize_executor
        )
        # The list of currently pending tasks. This is mainly used to keep
        # track of which tasks we need to wait for un shutdown.
        self.running_tasks = list[asyncio.Task]()
        # Reference to the main tasks, which pops from the queue and pushes
        # tasks to the worker processes.
        self.main_task: asyncio.Task[None] | None = None
        # We don't want to accept more tasks than we have workers, so we use a
        # semaphore to keep track of how many tasks we are currently
        # processing.
        self.available_workers = asyncio.Semaphore(2)

    async def run_task(self, spec: TaskSpec) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, spec.function)
        except BaseException as e:
            print(f"Task failed: {e!r}")
        finally:
            if task := asyncio.current_task():
                self.running_tasks.remove(task)
            self.available_workers.release()

    async def run_next_task(self) -> None:
        await self.available_workers.acquire()
        task = None
        spec = await self.queue.get()
        task = asyncio.create_task(self.run_task(spec))
        self.running_tasks.append(task)

    async def main(self) -> None:
        self.main_task = asyncio.current_task()
        print(self.main_task)
        try:
            while True:
                await self.run_next_task()
        except asyncio.CancelledError:
            print(f"Main task was cancelled, waiting for tasks to finish")
            if self.running_tasks:
                await asyncio.wait(
                    self.running_tasks, return_when=asyncio.ALL_COMPLETED
                )
            print(f"All tasks have finished, exiting")

    def run(self) -> None:
        self.queue.put_nowait(TaskSpec(test_task))
        self.queue.put_nowait(TaskSpec(test_task))
        self.queue.put_nowait(TaskSpec(test_task))
        with self.executor:
            asyncio.run(self.main())
