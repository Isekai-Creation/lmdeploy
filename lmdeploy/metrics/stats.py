# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from lmdeploy.messages import EngineEvent, EngineOutput, ResponseType, ScheduleMetrics


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler.

    Attributes:
        num_total_reqs: the number of all requests received since server start.
        num_finished_reqs: the number of successfully completed requests since server start.
        num_running_reqs: currently executing requests.
        num_waiting_reqs: Requests queued waiting for execution.
        gpu_cache_usage: Fraction of GPU KV blocks utilized (0.0 to 1.0).
    """

    num_total_reqs: int = 0
    num_finished_reqs: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    gpu_cache_usage: float = 0.0

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('SchedulerStats(\n'
                f'  num_total_reqs={self.num_total_reqs},\n'
                f'  num_finished_reqs={self.num_finished_reqs},\n'
                f'  num_running_reqs={self.num_running_reqs},\n'
                f'  num_waiting_reqs={self.num_waiting_reqs},\n'
                f'  gpu_cache_usage={self.gpu_cache_usage:.6f},\n'
                ')')

    def update_from_schedule_metrics(self, scheduled_metrics: ScheduleMetrics):
        self.num_running_reqs = scheduled_metrics.active_seqs
        self.num_waiting_reqs = scheduled_metrics.waiting_seqs
        self.gpu_cache_usage = 1.0 - (scheduled_metrics.free_blocks / scheduled_metrics.total_blocks)


class RequestState:
    """State of a request."""

    def __init__(self, arrival_time: float = None, prompt_tokens: int = 0):
        """Initialize the state of a request.

        Args:
            arrival_time (float, optional): The timestamp when the request arrives.
                If not provided, the current time will be used. Defaults to None.
            prompt_tokens (int, optional): The number of tokens in the prompt. Defaults to 0.
        """
        self.arrival_time = time.time() if arrival_time is None else arrival_time
        self.prompt_tokens = prompt_tokens

        # Number of tokens generated during the request inference.
        # It will be updated by IterationStats.update_from_output.
        self.generation_tokens: int = 0
        # Time when the request is put to the inference engine's queue. It will be updated according the EngineEvent
        self.queued_time: float = 0.0
        # Time when the request is scheduled to run. It will be updated according the EngineEvent
        self.scheduled_time: float = 0.0
        # Time when the first token is generated. It will be updated by IterationStats.update_from_output.
        self.first_token_time: float = 0.0
        # Time when the latest token is generated. It will be updated by IterationStats.update_from_output.
        self.lastest_token_time: float = 0.0
        # Time when a request finishes generation. It will be updated by IterationStats.update_from_output.
        self.finish_time: float = 0.0
        self.finish_reason: ResponseType = None

    def update_from_events(self, engine_events: List[EngineEvent]):
        # Avoid circular dependency
        from lmdeploy.messages import EventType

        for event in engine_events:
            if event.type == EventType.QUEUED:
                self.queued_time = event.timestamp
            elif event.type == EventType.SCHEDULED:
                if self.scheduled_time == 0.0:  # ignore preemptions
                    self.scheduled_time = event.timestamp
            # FIXME: deal with preempted case
            # elif event.type == EventType.PREEMPTED:
            #     self.num_preempted_reqs += 1

    @property
    def finish_stats(self) -> 'FinishedRequestStats':
        """Return stats of a finished request.

        It has to be called when a request is finished
        """

        e2e_latency = self.finish_time - self.arrival_time

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = self.scheduled_time - self.queued_time

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = self.first_token_time - self.scheduled_time

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = self.finish_time - self.first_token_time

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = self.finish_time - self.scheduled_time

        finished_req = \
            FinishedRequestStats(finish_reason=self.finish_reason,
                                 e2e_latency=e2e_latency,
                                 prompt_tokens=self.prompt_tokens,
                                 generation_tokens=self.generation_tokens,
                                 queued_time=queued_time,
                                 prefill_time=prefill_time,
                                 inference_time=inference_time,
                                 decode_time=decode_time)
        return finished_req

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('RequestState(\n'
                f'  arrival_time={self.arrival_time:.6f},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  generation_tokens={self.generation_tokens},\n'
                f'  queued_time={self.queued_time:.6f},\n'
                f'  scheduled_time={self.scheduled_time:.6f},\n'
                f'  first_token_time={self.first_token_time:.6f},\n'
                f'  latest_token_time={self.lastest_token_time:.6f},\n'
                ')')


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""
    finish_reason: ResponseType
    e2e_latency: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('FinishedRequestStats(\n'
                f'  e2e_latency={self.e2e_latency:.6f},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  generation_tokens={self.generation_tokens},\n'
                f'  queued_time={self.queued_time:.6f},\n'
                f'  prefill_time={self.prefill_time:.6f},\n'
                f'  inference_time={self.inference_time:.6f},\n'
                f'  decode_time={self.decode_time:.6f}\n'
                ')')


class IterationStats:
    """Stats associated with one token generation iteration of a request."""

    def __init__(self):
        # Record the timestamp when this iteration finished
        self.iteration_timestamp = time.time()
        # The number of newly generated tokens in this iteration
        self.new_generation_tokens = 0
        # The number of prompt tokens processed in this iteration
        self.prompt_tokens = 0
        # Time to First Token (TTFT), initialized as None and will be updated later
        self.ttft: Optional[float] = None
        # Time per Output Token (TPOT), initialized as None and will be updated later
        self.tpot: Optional[float] = None
        # Iter-Token Latency, initialized as None and will be updated later
        self.itl: Optional[float] = None

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('IterationStats(\n'
                f'  iteration_timestamp={self.iteration_timestamp:.6f},\n'
                f'  new_generation_tokens={self.new_generation_tokens},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  ttft={self.ttft},\n'
                f'  tpot={self.tpot},\n'
                f'  itl={self.itl},\n'
                ')')

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(self, outputs: EngineOutput, req_state: RequestState):
        """Update the iteration statistics based on the engine output and
        request state.

        Args:
            outputs (EngineOutput): The output from the engine containing information about the current iteration.
            req_state (RequestState): The state of the request, including timestamps and token counts.
        """
        if outputs.req_metrics is None:
            # when users visit "/abort_request" endpoint, `req_metrics` might be None
            return
        new_generation_tokens = len(outputs.token_ids)
        if new_generation_tokens == 0:
            return
        self.new_generation_tokens = new_generation_tokens
        if req_state.first_token_time == 0:
            # It means the first token is generated in this iteration
            req_state.first_token_time = outputs.req_metrics.token_timestamp
            self.prompt_tokens = req_state.prompt_tokens
            self.ttft = self._time_since(req_state.arrival_time)
        else:
            self.itl = self._time_since(req_state.lastest_token_time)
            self.tpot = self._time_since(req_state.lastest_token_time) / self.new_generation_tokens
        # update the latest token generation time
        req_state.lastest_token_time = outputs.req_metrics.token_timestamp
        # update the number of generated tokens
        req_state.generation_tokens += new_generation_tokens

        if outputs.status != ResponseType.SUCCESS:
            req_state.finish_reason = outputs.status
            req_state.finish_time = self.iteration_timestamp


# modify from vllm
@dataclass
class SpeculativeDecodingStats:
    """Speculative decoding stats."""

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: np.ndarray = None
    # Optional EAGLE3 target-tree metrics, populated when the backend
    # reports a nested ``tree_decode`` block in ``spec_info``.
    tree_num_draft_tokens: int = 0
    tree_num_target_tokens: int = 0
    tree_num_accepted_tokens: int = 0
    # Extended EAGLE3 metrics aggregated over drafts/steps.
    total_steps: int = 0
    total_tokens_per_seq: int = 0
    max_tokens_per_seq: int = 0
    total_accepted_len: int = 0
    max_accepted_len: int = 0
    steps_with_accept_ge2: int = 0
    total_committed_extras: int = 0

    def __post_init__(self):
        assert self.num_spec_tokens > 0
        self.num_accepted_tokens_per_pos = np.zeros(self.num_spec_tokens)

    def update_from_output(self, outputs: EngineOutput):
        """Update from engine output."""
        if spec_info := getattr(outputs.req_metrics, 'spec_info', None):
            num_drafts = int(spec_info.get("num_drafts", 0))
            if num_drafts <= 0:
                # Older builds exposed only per-request aggregates without a
                # draft-count. Treat the entire request as a single draft to
                # keep metrics bounded and non-crashing.
                num_drafts = 1

            num_draft_tokens = int(spec_info.get("num_draft_tokens", 0))
            num_accepted_tokens = int(spec_info.get("num_accepted_tokens", 0))

            # Basic draft/accept counts.
            self.num_drafts += num_drafts
            self.num_draft_tokens += num_draft_tokens
            self.num_accepted_tokens += num_accepted_tokens

            # Step-level aggregates are reported by TurboMind as totals across
            # drafts (per-request), so we aggregate using the draft count as a
            # weight.
            self.total_steps += num_drafts
            self.total_tokens_per_seq += num_draft_tokens
            self.total_accepted_len += num_accepted_tokens

            max_tokens_per_seq = int(spec_info.get("max_tokens_per_seq", 0))
            if max_tokens_per_seq <= 0 and num_drafts > 0:
                max_tokens_per_seq = int(num_draft_tokens / num_drafts) if num_drafts else 0
            self.max_tokens_per_seq = max(self.max_tokens_per_seq, max_tokens_per_seq)

            max_accepted_len_step = int(spec_info.get("max_accepted_len", 0))
            if max_accepted_len_step <= 0 and num_drafts > 0:
                max_accepted_len_step = int(num_accepted_tokens / num_drafts) if num_drafts else 0
            self.max_accepted_len = max(self.max_accepted_len, max_accepted_len_step)

            steps_accept_ge2 = int(spec_info.get("steps_accept_ge2", 0))
            self.steps_with_accept_ge2 += steps_accept_ge2

            total_committed_extras = int(spec_info.get("total_committed_extras", 0))
            self.total_committed_extras += total_committed_extras

            # Optional tree-aware metrics.
            tree_info = spec_info.get("tree_decode") if isinstance(spec_info, dict) else None
            if tree_info:
                self.tree_num_draft_tokens += int(tree_info.get("num_tree_draft_tokens", 0))
                self.tree_num_target_tokens += int(tree_info.get("num_tree_target_tokens", 0))
                self.tree_num_accepted_tokens += int(tree_info.get("num_tree_accepted_tokens", 0))

    def update_per_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        """Update with per draft stats."""
        if num_draft_tokens > 0:
            self.num_drafts += 1
            self.num_draft_tokens += num_draft_tokens
            self.num_accepted_tokens += num_accepted_tokens
            self.num_accepted_tokens_per_pos[:num_accepted_tokens] += 1

    def __repr__(self):
        """Return a human-readable string representation."""
        draft_acceptance_rate = (self.num_accepted_tokens / self.num_draft_tokens *
                                 100 if self.num_draft_tokens > 0 else float('nan'))

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (self.num_accepted_tokens /
                                      self.num_drafts) if self.num_drafts > 0 else float('nan')

        acceptance_rates = self.num_accepted_tokens_per_pos / self.num_drafts if self.num_drafts > 0 else [
            float('nan')
        ] * self.num_accepted_tokens
        rates_str = ', '.join(f'{p:.3f}' for p in acceptance_rates)

        return ('SpeculativeDecodingStats('
                f'num_spec_tokens={self.num_spec_tokens}, '
                f'num_drafts={self.num_drafts}, '
                f'num_draft_tokens={self.num_draft_tokens}, '
                f'num_accepted_tokens={self.num_accepted_tokens}, '
                f'draft_acceptance_rate={draft_acceptance_rate:.2f}%, '
                f'mean_acceptance_length={mean_acceptance_length:.2f}, '
                f'per_position_acceptance_rate={rates_str})')


@dataclass
class EagleMetricsSummary:
    """Aggregate view of EAGLE speculative decoding metrics.

    This wraps :class:`SpeculativeDecodingStats` into a compact summary
    suitable for logging or saving alongside benchmark results.
    """

    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    draft_acceptance_rate: float
    # Conventionally includes the bonus token:
    #   mean_acceptance_length = 1 + num_accepted_tokens / num_drafts.
    # This can be large for long runs; use mean_accepted_tokens_per_draft
    # for a more intuitive per-draft scalar.
    mean_acceptance_length: float
    # Additional context: how many speculative tokens we attempted per draft.
    num_spec_tokens: int | None = None
    # Extended summary fields derived from SpeculativeDecodingStats.
    mean_tokens_per_seq: float | None = None
    max_tokens_per_seq: int | None = None
    max_acceptance_length: int | None = None
    fraction_steps_accept_ge2: float | None = None
    mean_committed_extras: float | None = None
    # Optional tree-aware aggregates derived from SpeculativeDecodingStats.
    tree_num_draft_tokens: int | None = None
    tree_num_target_tokens: int | None = None
    tree_num_accepted_tokens: int | None = None
    # More intuitive scalar: average accepted tokens per draft (excluding the
    # bonus token), i.e. num_accepted_tokens / num_drafts.
    mean_accepted_tokens_per_draft: float | None = None

    @classmethod
    def from_stats(cls, stats: SpeculativeDecodingStats) -> "EagleMetricsSummary":
        """Construct a summary from :class:`SpeculativeDecodingStats`."""
        num_drafts = stats.num_drafts
        num_draft_tokens = stats.num_draft_tokens
        num_accepted_tokens = stats.num_accepted_tokens

        if num_draft_tokens > 0:
            draft_acceptance_rate = num_accepted_tokens / num_draft_tokens
        else:
            draft_acceptance_rate = float("nan")

        if num_drafts > 0:
            # Conventionally, mean acceptance length includes the bonus token.
            mean_acceptance_length = 1.0 + (num_accepted_tokens / num_drafts)
            mean_accepted_tokens_per_draft = num_accepted_tokens / num_drafts
        else:
            mean_acceptance_length = float("nan")
            mean_accepted_tokens_per_draft = float("nan")

        # Extended fields: averaged over speculative steps.
        if stats.total_steps > 0:
            mean_tokens_per_seq = stats.total_tokens_per_seq / stats.total_steps
            max_tokens_per_seq = stats.max_tokens_per_seq
            max_acceptance_length = stats.max_accepted_len
            fraction_steps_accept_ge2 = stats.steps_with_accept_ge2 / stats.total_steps
            mean_committed_extras = stats.total_committed_extras / stats.total_steps
        else:
            mean_tokens_per_seq = float("nan")
            max_tokens_per_seq = 0
            max_acceptance_length = 0
            fraction_steps_accept_ge2 = float("nan")
            mean_committed_extras = float("nan")

        tree_num_draft_tokens = stats.tree_num_draft_tokens or 0
        tree_num_target_tokens = stats.tree_num_target_tokens or 0
        tree_num_accepted_tokens = stats.tree_num_accepted_tokens or 0

        return cls(
            num_drafts=num_drafts,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            draft_acceptance_rate=draft_acceptance_rate,
            mean_acceptance_length=mean_acceptance_length,
            num_spec_tokens=stats.num_spec_tokens,
            mean_tokens_per_seq=mean_tokens_per_seq,
            max_tokens_per_seq=max_tokens_per_seq,
            max_acceptance_length=max_acceptance_length,
            fraction_steps_accept_ge2=fraction_steps_accept_ge2,
            mean_committed_extras=mean_committed_extras,
            tree_num_draft_tokens=tree_num_draft_tokens or None,
            tree_num_target_tokens=tree_num_target_tokens or None,
            tree_num_accepted_tokens=tree_num_accepted_tokens or None,
            mean_accepted_tokens_per_draft=mean_accepted_tokens_per_draft,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the summary."""
        out = {
            "num_drafts": int(self.num_drafts),
            "total_draft_tokens": int(self.num_draft_tokens),
            "total_accepted_tokens": int(self.num_accepted_tokens),
            "mean_acceptance_rate": float(self.draft_acceptance_rate),
            "mean_acceptance_length": float(self.mean_acceptance_length),
        }
        if self.num_spec_tokens is not None:
            out["num_spec_tokens"] = int(self.num_spec_tokens)
        if self.mean_accepted_tokens_per_draft is not None:
            out["mean_accepted_tokens_per_draft"] = float(self.mean_accepted_tokens_per_draft)
        # Extended fields are optional; include them when present.
        if self.mean_tokens_per_seq is not None:
            out["mean_tokens_per_seq"] = float(self.mean_tokens_per_seq)
        if self.max_tokens_per_seq is not None:
            out["max_tokens_per_seq"] = int(self.max_tokens_per_seq)
        if self.max_acceptance_length is not None:
            out["max_acceptance_length"] = int(self.max_acceptance_length)
        if self.fraction_steps_accept_ge2 is not None:
            out["fraction_steps_accept_ge2"] = float(self.fraction_steps_accept_ge2)
        if self.mean_committed_extras is not None:
            out["mean_committed_extras"] = float(self.mean_committed_extras)
        # Preserve the original schema and, when tree-aware metrics are
        # available, attach them under a nested ``tree_decode`` block.
        if self.tree_num_draft_tokens is not None or self.tree_num_target_tokens is not None \
                or self.tree_num_accepted_tokens is not None:
            out["tree_decode"] = {
                "total_tree_draft_tokens": int(self.tree_num_draft_tokens or 0),
                "total_tree_target_tokens": int(self.tree_num_target_tokens or 0),
                "total_tree_accepted_tokens": int(self.tree_num_accepted_tokens or 0),
            }
        return out
