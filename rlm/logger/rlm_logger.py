"""
Logger for RLM iterations.

Captures run metadata and iterations in memory so they can be attached to
RLMChatCompletion.metadata. Optionally writes the same data to JSON-lines files.
"""

import copy
import json
import os
import uuid
from datetime import datetime

from rlm.core.types import RLMIteration, RLMMetadata


class RLMLogger:
    """
    Captures trajectory (run metadata + iterations) for each completion.
    By default only captures in memory; set log_dir to also save to disk.

    - log_dir=None: trajectory is available via get_trajectory() and can be
      attached to RLMChatCompletion.metadata (no disk write).
    - log_dir="path": same capture plus appends to a JSONL file per run.
    - include_locals=False: omit diagnostic locals before serialization.
    - child_log_dir="path": save each child's full I/O in a separate durable file.
    - log_model_responses=True: also persist disk-only model I/O before code runs.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        file_name: str = "rlm",
        *,
        include_locals: bool = True,
        child_log_dir: str | None = None,
        helper_call_id: str | None = None,
        log_model_responses: bool = False,
    ):
        # Diagnostic policies only; none of these change live REPL state or feedback.
        self.include_locals = include_locals
        self.child_log_dir = child_log_dir
        self.helper_call_id = helper_call_id
        self.log_model_responses = log_model_responses
        self._save_to_disk = log_dir is not None
        self.log_dir = log_dir
        self.log_file_path: str | None = None
        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = str(uuid.uuid4())[:8]
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._run_metadata: dict | None = None
        self._iterations: list[dict] = []
        self._iteration_count = 0
        self._metadata_logged = False

    def child_logger(self, helper_call_id: str | None = None) -> "RLMLogger":
        """Give every child the same policy and optional flat child-only directory."""
        return RLMLogger(
            log_dir=self.child_log_dir,
            include_locals=self.include_locals,
            child_log_dir=self.child_log_dir,
            helper_call_id=helper_call_id,
            log_model_responses=self.log_model_responses,
        )

    def logging_metadata(self) -> dict:
        """Record nondefault diagnostic policy and link a child file to its helper."""
        fields = {}
        if not self.include_locals:
            fields["include_locals"] = False
        if self.log_model_responses:
            fields["log_model_responses"] = True
        if self.child_log_dir is not None:
            fields["child_log_dir"] = self.child_log_dir
            fields["log_file_path"] = self.log_file_path
        if self.helper_call_id is not None:
            fields["helper_call_id"] = self.helper_call_id
        return fields

    def write_entry(self, entry: dict, *, durable: bool = False) -> None:
        """Append one record; diagnostic child turns survive a client-process kill."""
        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as stream:
                json.dump(entry, stream)
                stream.write("\n")
                if durable or (
                    self.child_log_dir is not None and self.log_dir == self.child_log_dir
                ):
                    stream.flush()
                    os.fsync(stream.fileno())

    def log_model_response(self, iteration: RLMIteration) -> None:
        """Persist received model I/O before code runs, without counting an iteration."""
        if self.log_model_responses and self._save_to_disk:
            self.write_entry(
                {
                    "type": "model_response",
                    "iteration": self._iteration_count + 1,
                    "timestamp": datetime.now().isoformat(),
                    **iteration.to_dict(include_locals=False),
                },
                durable=True,
            )

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Capture run metadata (and optionally write to file)."""
        if self._metadata_logged:
            return

        self._run_metadata = copy.deepcopy({**metadata.to_dict(), **self.logging_metadata()})
        self._metadata_logged = True

        if self._save_to_disk and self.log_file_path:
            entry = {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **self._run_metadata,
            }
            self.write_entry(entry)

    def log(self, iteration: RLMIteration) -> None:
        """Capture one iteration (and optionally append to file)."""
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(include_locals=self.include_locals),
        }
        # Deferred child trajectories must retain the prompt/metadata as seen
        # at this turn, not references mutated by later history extensions.
        entry = copy.deepcopy(entry)
        self._iterations.append(entry)

        self.write_entry(entry)

    def clear_iterations(self) -> None:
        """Reset iterations for the next completion (trajectory is per completion)."""
        self._iterations = []
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        """Return captured run_metadata + iterations for the current completion, or None if no metadata yet."""
        if self._run_metadata is None:
            return None
        return {
            "run_metadata": self._run_metadata,
            "iterations": list(self._iterations),
        }

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
