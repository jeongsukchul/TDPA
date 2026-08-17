"""Causally aligned action-response sequence windows.

Temporal convention
-------------------
Index ``i`` denotes a sensor timestamp. ``action[i]`` is issued after observing
that timestamp, and the earliest response caused by it is ``response[i + 1]``.
For a sample anchored at ``t`` this dataset therefore returns:

* history observations/actions from ``[t-H, t)``;
* future actions from ``[t, t+F)``;
* future responses from ``[t+response_offset, t+response_offset+F)``.

The default response offset is one and cannot be zero, preventing a same-step
sensor value from being mislabeled as the effect of a not-yet-issued action.
Padding is zero-valued and every padded position is identified by a boolean
mask.  Privileged arrays and arbitrary episode metadata are never returned by
``__getitem__``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ArrayLike = np.ndarray | torch.Tensor


def _as_time_major_tensor(value: ArrayLike, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a tensor") from error
    if tensor.ndim < 1:
        raise ValueError(f"{name} must have a leading time dimension")
    if tensor.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{name} must contain real numeric values")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    # A private CPU copy makes episode contents stable even if a collector
    # reuses or mutates its source buffer after construction.
    return tensor.detach().cpu().clone().contiguous()


@dataclass(frozen=True)
class FixedFormatEpisode:
    """One episode with a fixed time-major deployment/target schema.

    Privileged information has a separate namespace by construction.  It may
    be used by a teacher dataset in a future module, but this deployment
    sequence dataset never emits it.
    """

    rgbd: ArrayLike
    proprio: ArrayLike
    actions: ArrayLike
    responses: ArrayLike
    privileged: Mapping[str, ArrayLike] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensors = {
            "rgbd": _as_time_major_tensor(self.rgbd, "rgbd"),
            "proprio": _as_time_major_tensor(self.proprio, "proprio"),
            "actions": _as_time_major_tensor(self.actions, "actions"),
            "responses": _as_time_major_tensor(self.responses, "responses"),
        }
        lengths = {name: value.shape[0] for name, value in tensors.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode fields must share one time dimension, got {lengths}")
        if tensors["proprio"].ndim != 2:
            raise ValueError("proprio must have shape [T, proprio_dim]")
        if tensors["actions"].ndim != 2:
            raise ValueError("actions must have shape [T, action_dim]")
        if tensors["responses"].ndim != 2:
            raise ValueError("responses must have shape [T, response_dim]")
        for name, value in tensors.items():
            object.__setattr__(self, name, value)

        if not isinstance(self.privileged, Mapping):
            raise TypeError("privileged must be a mapping")
        privileged: dict[str, torch.Tensor] = {}
        for key, value in self.privileged.items():
            if not isinstance(key, str) or not key:
                raise ValueError("privileged keys must be non-empty strings")
            tensor = _as_time_major_tensor(value, f"privileged[{key!r}]")
            if tensor.shape[0] != self.length:
                raise ValueError(
                    f"privileged[{key!r}] has length {tensor.shape[0]}, expected {self.length}"
                )
            privileged[key] = tensor
        object.__setattr__(self, "privileged", privileged)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class WindowAlignment:
    """Inspectable source indices for one dataset item; ``-1`` means padding."""

    episode_index: int
    anchor: int
    history_indices: tuple[int, ...]
    future_action_indices: tuple[int, ...]
    future_response_indices: tuple[int, ...]


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Slice fixed-format episodes into masked causal training windows."""

    def __init__(
        self,
        episodes: Sequence[FixedFormatEpisode],
        history_length: int,
        future_horizon: int,
        *,
        response_offset: int = 1,
        allow_partial_windows: bool = True,
        require_equal_episode_length: bool = True,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.history_length = _validate_positive_int(history_length, "history_length")
        self.future_horizon = _validate_positive_int(future_horizon, "future_horizon")
        self.response_offset = _validate_positive_int(response_offset, "response_offset")
        if not isinstance(allow_partial_windows, bool):
            raise TypeError("allow_partial_windows must be bool")
        if not isinstance(require_equal_episode_length, bool):
            raise TypeError("require_equal_episode_length must be bool")
        if not isinstance(output_dtype, torch.dtype) or not output_dtype.is_floating_point:
            raise TypeError("output_dtype must be a floating-point torch dtype")
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if not all(isinstance(episode, FixedFormatEpisode) for episode in episodes):
            raise TypeError("every episode must be a FixedFormatEpisode")

        self.episodes = tuple(episodes)
        self.allow_partial_windows = allow_partial_windows
        self.output_dtype = output_dtype
        self._validate_episode_schema(require_equal_episode_length)
        self._anchors = self._build_anchors()
        if not self._anchors:
            raise ValueError("no valid windows for the requested history/horizon settings")

    def _validate_episode_schema(self, require_equal_length: bool) -> None:
        first = self.episodes[0]
        expected_shapes = {
            "rgbd": tuple(first.rgbd.shape[1:]),
            "proprio": tuple(first.proprio.shape[1:]),
            "actions": tuple(first.actions.shape[1:]),
            "responses": tuple(first.responses.shape[1:]),
        }
        expected_privileged = {
            key: tuple(value.shape[1:]) for key, value in first.privileged.items()
        }
        for index, episode in enumerate(self.episodes[1:], start=1):
            shapes = {
                "rgbd": tuple(episode.rgbd.shape[1:]),
                "proprio": tuple(episode.proprio.shape[1:]),
                "actions": tuple(episode.actions.shape[1:]),
                "responses": tuple(episode.responses.shape[1:]),
            }
            if shapes != expected_shapes:
                raise ValueError(f"episode {index} schema {shapes} differs from {expected_shapes}")
            privileged = {key: tuple(value.shape[1:]) for key, value in episode.privileged.items()}
            if privileged != expected_privileged:
                raise ValueError("all episodes must use the same privileged schema")
            if require_equal_length and episode.length != first.length:
                raise ValueError("all episodes must have equal length in fixed-length mode")

    def _build_anchors(self) -> tuple[tuple[int, int], ...]:
        anchors: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            if self.allow_partial_windows:
                # Require at least one past element and one causal action-response pair.
                first_anchor = 1
                stop_anchor = episode.length - self.response_offset
            else:
                first_anchor = self.history_length
                stop_anchor = episode.length - self.response_offset - self.future_horizon + 1
            anchors.extend(
                (episode_index, anchor)
                for anchor in range(first_anchor, max(first_anchor, stop_anchor))
            )
        return tuple(anchors)

    def __len__(self) -> int:
        return len(self._anchors)

    @staticmethod
    def _indices(start: int, count: int, valid_low: int, valid_high: int) -> tuple[int, ...]:
        return tuple(
            index if valid_low <= index < valid_high else -1
            for index in range(start, start + count)
        )

    def window_alignment(self, item: int) -> WindowAlignment:
        episode_index, anchor = self._anchors[item]
        length = self.episodes[episode_index].length
        return WindowAlignment(
            episode_index=episode_index,
            anchor=anchor,
            history_indices=self._indices(
                anchor - self.history_length, self.history_length, 0, anchor
            ),
            future_action_indices=self._indices(
                anchor, self.future_horizon, anchor, length - self.response_offset
            ),
            future_response_indices=self._indices(
                anchor + self.response_offset,
                self.future_horizon,
                anchor + self.response_offset,
                length,
            ),
        )

    def _gather(
        self, source: torch.Tensor, indices: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (len(indices), *source.shape[1:])
        output = torch.zeros(shape, dtype=self.output_dtype)
        mask = torch.tensor([index >= 0 for index in indices], dtype=torch.bool)
        valid_positions = mask.nonzero(as_tuple=False).flatten()
        if valid_positions.numel():
            source_indices = torch.tensor(
                [index for index in indices if index >= 0], dtype=torch.long
            )
            output[valid_positions] = source.index_select(0, source_indices).to(
                dtype=self.output_dtype
            )
        return output, mask

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        alignment = self.window_alignment(item)
        episode = self.episodes[alignment.episode_index]
        rgbd_history, history_mask = self._gather(episode.rgbd, alignment.history_indices)
        proprio_history, proprio_mask = self._gather(episode.proprio, alignment.history_indices)
        action_history, action_history_mask = self._gather(
            episode.actions, alignment.history_indices
        )
        if not (
            torch.equal(history_mask, proprio_mask)
            and torch.equal(history_mask, action_history_mask)
        ):  # pragma: no cover - one shared index tuple makes this invariant structural
            raise AssertionError("history modality masks diverged")

        future_actions, future_action_mask = self._gather(
            episode.actions, alignment.future_action_indices
        )
        future_responses, future_response_mask = self._gather(
            episode.responses, alignment.future_response_indices
        )
        return {
            "rgbd_history": rgbd_history,
            "proprio_history": proprio_history,
            "action_history": action_history,
            "history_mask": history_mask,
            "future_action_sequence": future_actions,
            "future_response_sequence": future_responses,
            "future_action_mask": future_action_mask,
            "future_response_mask": future_response_mask,
            "future_mask": future_action_mask & future_response_mask,
        }
