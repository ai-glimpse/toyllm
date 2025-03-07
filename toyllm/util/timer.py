import time
import types

# Constants for time conversion
_MICROSECOND_THRESHOLD = 0.001  # 1 millisecond in seconds
_MILLISECOND_THRESHOLD = 1.0  # 1 second


class Timer:
    """A context manager for timing code execution with rich output."""

    def __init__(self, name: str = "Code Block") -> None:
        """Initialize the Timer.

        Args:
            name (str): Name of the code block being timed.
        """
        self.name: str = name
        self.elapsed: float = 0.0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer when entering the context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop the timer when exiting the context."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self._print_result()

    def _print_result(self) -> None:
        """Print the execution time using rich formatting."""
        # Choose a proper time unit based on magnitude
        if self.elapsed < _MICROSECOND_THRESHOLD:  # less than 1ms
            time_str = f"{self.elapsed * 1_000_000:.2f} μs"
        elif self.elapsed < _MILLISECOND_THRESHOLD:  # less than 1s
            time_str = f"{self.elapsed * 1_000:.2f} ms"
        else:
            time_str = f"{self.elapsed:.4f} s"

        print(f"{self.name} completed in {time_str}")
