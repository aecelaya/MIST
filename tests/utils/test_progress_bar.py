"""Unit tests for MIST training/validation progress bars."""
from typing import Any
from unittest.mock import patch
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# MIST imports.
from mist.utils.progress_bar import (
    TrainProgressBar, ValidationProgressBar, get_progress_bar
)


class SpyProgress:
    """Spy replacement for rich.progress.Progress used for tests.

    Captures constructor columns, add_task calls, update calls, and start/stop
    state without performing any terminal output.
    """

    def __init__(self, *columns: Any):
        self.columns = columns
        self.started = False
        self.stopped = False
        self._next_task_id = 1
        self.tasks: list[int] = []
        self.add_task_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    # API surface used by the module under test.
    def add_task(self, description: str, total: int, **fields: Any) -> int:
        """Add a task with description, total, and custom fields."""
        task_id = self._next_task_id
        self._next_task_id += 1
        self.tasks.append(task_id)
        self.add_task_calls.append(
            {"description": description, "total": total, "fields": fields}
        )
        return task_id

    def update(self, task_id: int, **kwargs: Any) -> None:
        """Update a task with the given task_id and additional fields."""
        self.update_calls.append({"task_id": task_id, **kwargs})

    def start(self) -> None:
        """Start the progress bar (no-op for spy)."""
        self.started = True

    def stop(self) -> None:
        """Stop the progress bar (no-op for spy)."""
        self.stopped = True


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_initialization_creates_task():
    """TrainProgressBar should create a task with loss/lr fields initialized."""
    current_epoch = 3
    fold = 2
    epochs = 10
    train_steps = 123

    pb = TrainProgressBar(
        current_epoch=current_epoch,
        fold=fold,
        epochs=epochs,
        train_steps=train_steps,
    )

    # Ensure inner progress is our spy and a task was added.
    assert isinstance(pb.progress, SpyProgress)
    assert len(pb.progress.add_task_calls) == 1

    call = pb.progress.add_task_calls[0]
    assert call["description"] == "Training (loss)"
    assert call["total"] == train_steps
    # Initial custom fields.
    assert call["fields"]["loss"] == "loss: "
    assert call["fields"]["lr"] == "lr: "

    # Sanity check: columns were provided.
    assert len(pb.progress.columns) > 0


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_update_formats_loss_and_lr():
    """Update should advance by 1 and format loss and lr as expected."""
    pb = TrainProgressBar(current_epoch=1, fold=0, epochs=5, train_steps=10)

    loss = 0.123456
    lr = 1e-4
    pb.update(loss=loss, lr=lr)

    assert len(pb.progress.update_calls) == 1
    update = pb.progress.update_calls[0]

    # Advance step.
    assert update["advance"] == 1

    # Loss formatting: 4 decimals.
    assert update["loss"] == f"loss: {loss:.4f}"

    # LR formatting: scientific with precision=3 (NumPy).
    expected_lr = f"lr: {np.format_float_scientific(lr, precision=3)}"
    assert update["lr"] == expected_lr


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_multiple_updates():
    """Multiple updates should each advance by 1 and record formatted fields."""
    pb = TrainProgressBar(current_epoch=1, fold=0, epochs=5, train_steps=3)

    values = [(0.2, 1e-3), (0.19, 9.5e-4), (0.181, 8.7e-4)]
    for loss, lr in values:
        pb.update(loss=loss, lr=lr)

    assert len(pb.progress.update_calls) == len(values)

    for (loss, lr), update in zip(values, pb.progress.update_calls):
        assert update["advance"] == 1
        assert update["loss"] == f"loss: {loss:.4f}"
        assert (
            update["lr"] == f"lr: {np.format_float_scientific(lr, precision=3)}"
        )


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_train_progressbar_context_manager_starts_and_stops():
    """Context manager should start on enter and stop on exit."""
    pb = TrainProgressBar(current_epoch=1, fold=0, epochs=5, train_steps=1)

    # Before context.
    assert not pb.progress.started
    assert not pb.progress.stopped

    with pb as _ctx:
        assert _ctx is pb
        assert pb.progress.started
        assert not pb.progress.stopped

    # After context.
    assert pb.progress.stopped


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_initialization_creates_task():
    """ValidationProgressBar should create a task with val_loss field."""
    val_steps = 42
    pb = ValidationProgressBar(val_steps=val_steps)

    assert isinstance(pb.progress, SpyProgress)
    assert len(pb.progress.add_task_calls) == 1

    call = pb.progress.add_task_calls[0]
    assert call["description"] == "Validation"
    assert call["total"] == val_steps
    assert call["fields"]["loss"] == "val_loss: "

    # Ensure columns were passed.
    assert len(pb.progress.columns) > 0


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_update_formats_loss():
    """Update should advance by 1 and format validation loss with 4 decimals."""
    pb = ValidationProgressBar(val_steps=5)

    loss = 0.987654
    pb.update(loss=loss)

    assert len(pb.progress.update_calls) == 1
    update = pb.progress.update_calls[0]
    assert update["advance"] == 1
    assert update["loss"] == f"val_loss: {loss:.4f}"


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_multiple_updates():
    """Multiple updates should each advance and format the loss correctly."""
    pb = ValidationProgressBar(val_steps=3)
    losses = [0.33, 0.31, 0.3051]

    for val in losses:
        pb.update(loss=val)

    assert len(pb.progress.update_calls) == len(losses)
    for val, update in zip(losses, pb.progress.update_calls):
        assert update["advance"] == 1
        assert update["loss"] == f"val_loss: {val:.4f}"


@patch("mist.utils.progress_bar.Progress", new=SpyProgress)
def test_validation_progressbar_context_manager_starts_and_stops():
    """Context manager should start on enter and stop on exit."""
    pb = ValidationProgressBar(val_steps=2)

    assert not pb.progress.started
    assert not pb.progress.stopped

    with pb as _ctx:
        assert _ctx is pb
        assert pb.progress.started
        assert not pb.progress.stopped

    assert pb.progress.stopped


def test_get_progress_bar_structure_and_usage():
    """get_progress_bar returns a Progress with expected columns and works."""
    name = "Unit Test Task"
    pb = get_progress_bar(name)

    # Basic type.
    assert isinstance(pb, Progress)

    # Verify the configured columns:
    # [Spinner, TextColumn, Bar, MofN, TaskProgress, TimeElapsed, TimeRemaining]
    cols = pb.columns
    assert len(cols) == 7
    assert isinstance(cols[0], SpinnerColumn)
    assert isinstance(cols[1], TextColumn)
    assert isinstance(cols[2], BarColumn)
    assert isinstance(cols[3], MofNCompleteColumn)
    assert isinstance(cols[4], TaskProgressColumn)
    assert isinstance(cols[5], TimeElapsedColumn)
    assert isinstance(cols[6], TimeRemainingColumn)

    # Ensure it behaves as a context manager and can run a task.
    # This should not raise and should allow advancing to completion.
    with pb as progress:
        task_id = progress.add_task(name, total=2)
        progress.advance(task_id)
        progress.advance(task_id)  # complete the task

        # Sanity: the task is finished.
        # Progress keeps task data in _tasks; use the public API where possible.
        # `tasks` is public in rich>=13.x; fallback: ensure no exception is raised.
        tasks = [t for t in progress.tasks if t.id == task_id]
        assert tasks and tasks[0].finished
