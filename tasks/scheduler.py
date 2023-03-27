import concurrent.futures
import os
import asyncio
import time
import dataclasses
import signal
from types import FrameType
from typing import Any, Callable


@dataclasses.dataclass()
class TaskSpec:
    function: Callable[..., Any]


def test_task() -> None:
    pid = os.getpid()
    print(f"Worker process ({pid}) is running a task")
    t0 = time.perf_counter()
    try:
        time.sleep(10)
    finally:
        t1 = time.perf_counter()
        print(
            f"Worker process ({pid}) finished running a task in {t1 - t0:.2f} seconds"
        )


num_sigints = 0


def handle_sigint_in_executor(_signal: int, _frame: FrameType | None) -> None:
    pid = os.getpid()

    global num_sigints

    num_sigints += 1
    if num_sigints > 1:
        print(
            f"Worker process ({pid}) received more than one sigint, "
            f"exiting immediately"
        )
        raise KeyboardInterrupt


def initialize_executor() -> None:
    # Ignore SIGINT in the executors. We want to have control over the
    # shutdown, rather than have the workers exit at unexpected times.
    signal.signal(signal.SIGINT, handle_sigint_in_executor)
    # Log a message
    pid = os.getpid()
    print(f"Worker process ({pid}) started")


class Scheduler:
    def __init__(self, *, num_processes: int = 2) -> None:
        # Queue for pending tasks. To be replaced with an external task queue
        self.queue = asyncio.Queue[TaskSpec]()
        # A process pool that we run tasks in
        self.sync_executor = concurrent.futures.ProcessPoolExecutor(
            initializer=initialize_executor,
            max_workers=num_processes,
        )
        # The list of currently pending tasks. This is mainly used to keep
        # track of which tasks we need to wait for un shutdown.
        self.running_tasks = set[asyncio.Task]()
        # We don't want to accept more tasks than we have workers, so we use a
        # semaphore to keep track of how many tasks we are currently
        # processing.
        self.available_workers = asyncio.Semaphore(num_processes)

    async def run_task(self, spec: TaskSpec) -> None:
        """
        Run a task. This pushed the task to a worker in the process pool and
        waits until it's done (or crashes).
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.sync_executor, spec.function)
        finally:
            self.available_workers.release()

    async def run_next_task(self) -> None:
        """
        Run the next queued task.

        This first acquires a free worker process, then fetches a task from the
        queue and starts it. Note that this does not wait until the task is
        completed.
        """

        await self.available_workers.acquire()
        try:
            spec = await self.queue.get()
        except Exception:
            self.available_workers.release()
            raise

        task = asyncio.create_task(self.run_task(spec))
        self.running_tasks.add(task)
        task.add_done_callback(self.running_tasks.discard)

    async def main(self) -> None:
        """
        Main run loop. This will run a loop, fetching tasks and executing them.
        """

        for key, process in self.sync_executor._processes.items():
            print(f"Process {key}: {process}")

        try:
            while True:
                await self.run_next_task()
        except asyncio.CancelledError:
            print("Shutdown requested")
            if self.running_tasks:
                print("Waiting for running tasks to finish")
                await asyncio.wait(
                    self.running_tasks, return_when=asyncio.ALL_COMPLETED
                )
                print("All running tasks have finished, exiting")

    def run(self) -> None:
        """
        Sync entrypoint. Does process pool cleanup.
        """

        self.queue.put_nowait(TaskSpec(test_task))
        self.queue.put_nowait(TaskSpec(test_task))
        self.queue.put_nowait(TaskSpec(test_task))
        with self.sync_executor:
            asyncio.run(self.main())
