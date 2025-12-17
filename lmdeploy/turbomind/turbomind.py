# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import base64
import copy
import json
import math
import os
import os.path as osp
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from multiprocessing.reduction import ForkingPickler
from queue import Queue
from typing import Any, Dict, List, Optional, Union
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

import lmdeploy
from lmdeploy.messages import (
    EngineOutput,
    GenerationConfig,
    ResponseType,
    ScheduleMetrics,
    TurbomindEngineConfig,
    DriftEngineConfig,
)
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger, get_max_batch_size, get_model

from .deploy.config import TurbomindModelConfig
from .supported_models import is_supported

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
lib_dir = osp.join(lmdeploy_dir, "lib")
if lib_dir not in sys.path:
    # Prepend so that the freshly built extension in lmdeploy/lib
    # takes precedence over any older _turbomind modules on sys.path.
    sys.path.insert(0, lib_dir)
import _turbomind as _tm  # noqa: E402
import _xgrammar as _xgr  # noqa: E402


from .tokenizer_info import TokenizerInfo  # noqa: E402

logger = get_logger("lmdeploy")

MAX_LOGPROBS = 1024


def _construct_stop_or_bad_words(words: List[int] = None):
    if words is None or len(words) == 0:
        return None
    offsets = list(range(1, len(words) + 1))
    combined = [words, offsets]
    return combined


def _np_dict_to_tm_dict(np_dict: dict):
    """Map numpy.ndarray to turbomind's tensor."""
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)

    return ret


def _tm_dict_to_torch_dict(tm_dict):
    """Map turbomind's tensor to torch's tensor."""
    ret = dict()
    for k, v in tm_dict.items():
        # If the underlying tensor uses uint32, remap to int32 for
        # PyTorch. In newer bindings we no longer expose _tm.DataType,
        # so rely on the dtype string when available.
        dtype = getattr(v, "dtype", None)
        if isinstance(dtype, str) and dtype.lower() in ("uint32", "uint32_t", "u32"):
            v = v.view("int32")
        ret[k] = torch.from_dlpack(v)

    return ret


def complete_parallel_config(cfg: TurbomindEngineConfig):
    if any(
        (
            cfg.attn_dp_size,
            cfg.attn_tp_size,
            cfg.mlp_dp_size,
            cfg.mlp_tp_size,
            cfg.outer_dp_size,
        )
    ):
        cfg.attn_dp_size = cfg.attn_dp_size or 1
        cfg.attn_tp_size = cfg.attn_tp_size or 1
        cfg.mlp_dp_size = cfg.mlp_dp_size or 1
        cfg.mlp_tp_size = cfg.mlp_tp_size or 1
        cfg.outer_dp_size = cfg.outer_dp_size or 1
        gcd = math.gcd(cfg.mlp_dp_size, cfg.attn_dp_size)
        cfg.outer_dp_size *= gcd
        cfg.mlp_dp_size //= gcd
        cfg.attn_dp_size //= gcd
        return True
    return False


def update_parallel_config(cfg: TurbomindEngineConfig):
    if not complete_parallel_config(cfg):
        total = cfg.dp * cfg.tp
        if not cfg.device_num:
            count = torch.cuda.device_count()
            if count == 0:
                # NVML failed - fallback to 1 GPU if CUDA is available
                if torch.cuda.is_available():
                    logger.warning("NVML initialization failed, assuming 1 GPU")
                    count = 1
                else:
                    raise RuntimeError(
                        "No CUDA devices detected. Ensure NVIDIA drivers are properly installed."
                    )
            if total < count:
                count = total
            cfg.device_num = count
        assert (
            total % cfg.device_num == 0
        ), f"total ({total}) must be divisible by device_num ({cfg.device_num})"
        overlap = total // cfg.device_num
        attn_dp_size = overlap
        mlp_tp_size = overlap
        inner_tp_size = cfg.tp // mlp_tp_size
        cfg.outer_dp_size = cfg.dp // attn_dp_size
        cfg.attn_dp_size = attn_dp_size
        cfg.attn_tp_size = inner_tp_size // cfg.cp
        cfg.attn_cp_size = cfg.cp
        cfg.mlp_dp_size = 1
        cfg.mlp_tp_size = mlp_tp_size * inner_tp_size
    assert (
        cfg.attn_dp_size * cfg.attn_tp_size * cfg.attn_cp_size
        == cfg.mlp_dp_size * cfg.mlp_tp_size
    )
    assert (
        cfg.attn_dp_size * cfg.attn_tp_size * cfg.attn_cp_size * cfg.outer_dp_size
        == cfg.device_num
    )
    cfg.devices = cfg.devices or list(range(cfg.device_num))


class TurboMind:
    """LMDeploy's inference engine.

    Args:
        model_path (str): the path of turbomind's model
        mode_name (str): the name of the served model
        chat_template_name (str): the name of the chat template, which is
            supposed to be a builtin chat template defined in
            `lmdeploy/model.py`
        engine_config (TurbomindEngineConfig): the config of the inference
            engine
        model_source (int): the source of the model, which is either
            turbomind model, or a transformers model
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = None,
        chat_template_name: str = None,
        engine_config: Union[TurbomindEngineConfig, DriftEngineConfig] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.chat_template_name = chat_template_name

        _engine_config = copy.deepcopy(engine_config)
        if _engine_config is None:
            _engine_config = TurbomindEngineConfig()

        # Optional override to force prefix caching on without requiring
        # every caller (e.g. benchmark scripts) to pass the flag explicitly.
        # This respects the user request to enable prefix caching globally
        # while avoiding direct modifications to benchmark_speculative.py.
        force_prefix_env = os.getenv("TM_FORCE_PREFIX_CACHING", "").strip().lower()
        if force_prefix_env in ("1", "true", "yes", "on"):
            _engine_config.enable_prefix_caching = True

        # Optional override for KV cache memory fraction. This allows
        # callers (or wrapper scripts) to cap TurboMind's KV cache
        # usage without modifying benchmark scripts. When set to a
        # float in (0, 1], it is treated as a fraction of free GPU
        # memory reserved for KV cache (e.g. 0.75 -> 75%).
        cache_frac_env = os.getenv("TM_CACHE_MAX_ENTRY_COUNT", "").strip()
        if cache_frac_env:
            try:
                cache_frac = float(cache_frac_env)
                if cache_frac > 0:
                    _engine_config.cache_max_entry_count = cache_frac
            except ValueError:
                # Ignore malformed overrides and keep the existing value.
                pass

        if _engine_config.max_batch_size is None:
            _engine_config.max_batch_size = get_max_batch_size("cuda")
        assert _engine_config.max_batch_size > 0, (
            "max_batch_size should be"
            f" greater than 0, but got {_engine_config.max_batch_size}"
        )

        # Handle DriftEngine config for new high-performance engine
        if isinstance(_engine_config, DriftEngineConfig):
            self._engine_type = "drift"
            self._setup_drift_engine(_engine_config, model_path)
            self.gpu_count = getattr(_engine_config, "device_num", 1) or 1
            self.devices = getattr(_engine_config, "devices", None) or list(range(self.gpu_count))
            self._engine_created = False
            return
        else:
            self._engine_type = "turbomind"
            update_parallel_config(_engine_config)

        self.gpu_count = _engine_config.device_num
        self.devices = _engine_config.devices
        self._engine_created = False

    def _setup_drift_engine(self, engine_config: DriftEngineConfig, model_path: str):
        """Setup DriftEngine-specific initialization.

        For HF-style GPT-OSS-120B paths (e.g. models/gpt-oss-120b), reuse the
        same HF→TurboMind conversion path as the legacy TurboMind engine
        (_from_hf + get_tm_model) so DriftEngine always binds a real
        LlamaBatch backed by converted TurboMind weights. No pre-exported
        \"-turbomind\" checkpoint is required.
        """
        from lmdeploy.messages import TurbomindEngineConfig

        # Enforce drift v1 constraints.
        engine_config.enable_speculative_decoding = False
        engine_config.enable_metrics = False
        if not engine_config.model_path:
            engine_config.model_path = model_path

        # Default drift dtype to BF16 for model weights/compute unless the
        # user explicitly requested a different dtype.
        if getattr(engine_config, "dtype", None) in (None, "", "auto", "float16", "fp16"):
            engine_config.dtype = "bf16"

        self._drift_config = engine_config

        from lmdeploy.config.drift_config import to_cpp_drift_engine_config
        cpp_args = to_cpp_drift_engine_config(engine_config)
        self._drift_cpp_args = cpp_args

        vocab_size = 0
        try:
            tok = Tokenizer(engine_config.model_path)
            self.tokenizer = tok
            vocab_size = getattr(tok, "vocab_size", 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Tokenizer init failed for drift backend: {exc}")
            self.tokenizer = None

        logger.info(f"DriftEngine backend requested: {engine_config}")
        logger.info(f"Converted to C++ args: {list(cpp_args.keys())}")

        # Build a TurboMindEngineConfig for HF conversion, mirroring the
        # core session / batch / dtype / cache knobs from the drift
        # config so that get_tm_model and _from_hf behave identically to
        # the legacy TurboMind engine.
        tm_engine = TurbomindEngineConfig()
        tm_engine.tp = engine_config.tp
        tm_engine.dp = 1
        tm_engine.cp = 1
        tm_engine.session_len = engine_config.session_len
        tm_engine.max_batch_size = engine_config.max_batch_size

        # Use BF16 weights/compute by default for GPT-OSS-120B; allow the
        # converter to override dtype when the hardware or model config
        # requires a fallback.
        tm_engine.dtype = "bfloat16"

        # Derive cache_max_entry_count from TM_CACHE_MAX_ENTRY_COUNT when
        # provided, otherwise reuse the drift config or fall back to 0.75.
        cache_frac_env = os.getenv("TM_CACHE_MAX_ENTRY_COUNT", "").strip()
        cache_frac = None
        if cache_frac_env:
            try:
                cache_frac = float(cache_frac_env)
            except ValueError:
                cache_frac = None
        if cache_frac is None:
            cache_frac = getattr(engine_config, "cache_max_entry_count", 0.75)
        tm_engine.cache_max_entry_count = max(0.1, min(cache_frac, 0.95))

        # Let the standard parallel config helper fill in device_num and
        # devices for TP=1 drift.
        update_parallel_config(tm_engine)
        self.gpu_count = tm_engine.device_num
        self.devices = tm_engine.devices

        # Run the same HF→TM conversion path as TurboMind, which sets
        # self._tm_model, self.config, self.engine_config (tm_engine),
        # and self.config_dict, then create the LlamaTritonModel.
        self.model_comm = self._from_hf(engine_config.model_path, tm_engine)

        # Drift-specific config used on the Python side for generation.
        # Reuse the TurboMind model config but override weight_type to
        # reflect the drift dtype.
        model_cfg = SimpleNamespace(vocab_size=vocab_size)
        weight_type = getattr(self.config.model_config, "weight_type", "bfloat16")
        self.config = SimpleNamespace(model_config=model_cfg, weight_type=weight_type)
        self.engine_config = engine_config

        self.node_id = 0
        self.node_num = 1
        self.session_len = engine_config.session_len
        self._drift_engine = None

    def _ensure_drift_engine(self):
        if self._drift_engine is None:
            self._create_drift_engine()

    def _create_drift_engine(self):
        """Create DriftEngine using C++ factory and start worker loop."""
        args = self._drift_cpp_args
        cfg = self._drift_config
        # Apply log level early for TM logger if provided
        try:
            import os
            if cfg and getattr(cfg, "log_level", None):
                os.environ.setdefault("TM_LOG_LEVEL", str(cfg.log_level))
        except Exception:
            pass
        try:
            # Prefer the path that reuses an existing LlamaTritonModel so
            # DriftEngine can bind a real LlamaBatch executor with GPU
            # weights, falling back to the dict-only factory when no
            # model_comm is available.
            engine = None
            if hasattr(_tm, "create_engine_with_model") and getattr(self, "model_comm", None) is not None:
                engine = _tm.create_engine_with_model(
                    self.model_comm,
                    cfg.model_path,
                    args,
                    backend="drift",
                )
            elif hasattr(_tm, "create_engine"):
                engine = _tm.create_engine(cfg.model_path, args, backend="drift")
            elif hasattr(_tm, "create_drift_engine"):
                # Fallback for builds exposing only create_drift_engine signature.
                sched = cfg.scheduler
                kv = cfg.kv
                engine = _tm.create_drift_engine(
                    cfg.model_path,
                    cfg.tp,
                    cfg.pp,
                    cfg.session_len,
                    cfg.max_batch_size,
                    cfg.dtype,
                    sched.max_num_batched_tokens,
                    sched.max_num_seqs,
                    sched.enable_chunked_prefill,
                    sched.max_num_partial_prefills,
                    sched.long_prefill_token_threshold,
                    sched.prefer_decode_over_prefill,
                    kv.kv_page_size,
                    kv.kv_capacity_bytes or 0,
                    kv.prefix_cache_enabled if hasattr(kv, "prefix_cache_enabled") else cfg.enable_prefix_caching,
                    cfg.prefer_high_throughput,
                    cfg.target_latency_ms_p50,
                    cfg.target_latency_ms_p95,
                    cfg.max_queued_requests,
                    cfg.abort_on_oom,
                    cfg.log_level,
                )
            else:
                raise AttributeError("No drift engine factory available in _turbomind")

            if engine is None:
                raise RuntimeError("DriftEngine factory did not return an engine instance")

            # Ensure the worker loop is running exactly once. The C++
            # wrappers guard start() internally, so it is safe to call
            # start() here even if the factory already started the
            # worker thread.
            should_start = True
            try:
                if hasattr(engine, "is_running") and engine.is_running():
                    should_start = False
            except Exception:
                should_start = True
            if should_start and hasattr(engine, "start"):
                engine.start()

            self._drift_engine = engine
            self._engine_created = True
            logger.info("DriftEngine created and worker started")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to create DriftEngine: {e}")
            raise

        self.session_len = self._drift_config.session_len
        self.speculative_manager = None

    def _check_unloaded_tm_params(self):
        tm_params = self._tm_model.tm_params
        if len(tm_params) > 0:
            uninitialized = list(tm_params.keys())
            logger.warning(
                "the model may not be loaded successfully "
                f"with {len(tm_params)} uninitialized params:\n{uninitialized}"
            )

    def _load_weights(self):
        """Load weights."""
        self._get_model_params()

        with torch.cuda.device(self.devices[0]):
            self._tm_model.export()

        self._check_unloaded_tm_params()

    def _process_weights(self):
        """Process weight."""
        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [
                self.node_id * self.gpu_count + device_id
                for device_id in range(self.gpu_count)
            ]
            for _ in e.map(
                self.model_comm.process_weight, range(self.gpu_count), ranks
            ):
                pass

    def _create_engine(self):
        """Create engine."""
        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [
                self.node_id * self.gpu_count + device_id
                for device_id in range(self.gpu_count)
            ]
            for _ in e.map(self.model_comm.create_engine, range(self.gpu_count), ranks):
                pass
        self._engine_created = True

    def _create_weight(self, model_comm):
        """Allocate weight buffer, load params if from_workspace."""

        # TODO: support mpi
        self.node_id = 0
        self.node_num = 1
        torch.cuda.synchronize()

        # create weight
        def _create_weight_func(device_id):
            rank = self.node_id * self.gpu_count + device_id
            model_comm.create_shared_weights(device_id, rank)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_create_weight_func, device_id))
            for future in futures:
                future.result()

    def _get_model_params(self):
        """Get turbomind model params when loading from hf."""

        model_comm = self.model_comm
        tm_params = self._tm_model.tm_params
        tm_params.clear()

        def _get_params(device_id, que):
            rank = self.node_id * self.gpu_count + device_id
            out = model_comm.get_params(device_id, rank)
            que.put(out)

        que = Queue()
        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_get_params, device_id, que))
            for future in futures:
                future.result()

        for _ in range(self.gpu_count):
            tensor_map = que.get()
            for k, v in tensor_map.items():
                if k not in tm_params:
                    tm_params[k] = [v]
                else:
                    tm_params[k].append(v)
        logger.warning(f"get {len(tm_params)} model params")

    def _postprocess_config(
        self, tm_config: TurbomindModelConfig, engine_config: TurbomindEngineConfig
    ):
        """Postprocess turbomind config by."""
        import copy

        self.config = copy.deepcopy(tm_config)
        # Update the attribute values in `self.config` with the valid values
        # from the corresponding attributes in `engine_config`, such as
        # `session_len`, `quant_policy`, `rope_scaling_factor`, etc.
        self.config.update_from_engine_config(engine_config)

        # update some attributes of `engine_config` which depends on
        # `session_len`
        self.engine_config = engine_config
        if (
            engine_config.max_prefill_token_num is not None
            and engine_config.num_tokens_per_iter == 0
        ):
            self.engine_config.num_tokens_per_iter = engine_config.max_prefill_token_num
            self.engine_config.max_prefill_iters = (
                self.config.session_len + engine_config.max_prefill_token_num - 1
            ) // engine_config.max_prefill_token_num

        # pack `self.config` and `self.engine_config` into a dict
        self.config_dict = self.config.to_dict()
        self.config_dict.update(dict(engine_config=asdict(self.engine_config)))
        logger.info(
            f"turbomind model config:\n\n" f"{json.dumps(self.config_dict, indent=2)}"
        )

    def _from_hf(self, model_path: str, engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert is_supported(model_path), (
            f"turbomind does not support {model_path}. "
            "Plz try pytorch engine instead."
        )

        # convert transformers model into turbomind model
        from .deploy.converter import get_tm_model

        tm_model = get_tm_model(
            model_path, self.model_name, self.chat_template_name, engine_config
        )

        self._postprocess_config(tm_model.tm_config, engine_config)

        if _tm is None:
            raise RuntimeError(
                "lmdeploy.turbomind.turbomind: _turbomind extension is None. "
                "This indicates the C++/CUDA extension failed to import; "
                "check that lmdeploy.lib._turbomind imports correctly in this environment."
            )
        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir="",
            config=yaml.safe_dump(self.config_dict),
            weight_type=self.config.model_config.weight_type,
        )

        # create empty weight
        self._create_weight(model_comm)
        # output model
        self._tm_model = tm_model
        return model_comm

    def sleep(self, level: int = 1):
        """Sleep the model."""
        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            for _ in e.map(
                self.model_comm.sleep, range(self.gpu_count), [level] * self.gpu_count
            ):
                pass

    def wakeup(self, tags: Optional[list[str]] = None):
        """Wakeup the model."""
        if tags is None:
            tags = ["weights", "kv_cache"]
        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [
                self.node_id * self.gpu_count + device_id
                for device_id in range(self.gpu_count)
            ]
            for _ in e.map(
                self.model_comm.wakeup,
                range(self.gpu_count),
                [tags] * self.gpu_count,
                ranks,
            ):
                pass

    def update_params(self, request: UpdateParamsRequest):
        """Update params.

        When using the this function, you need to set empty_init=True when creating the engine.

        For each request, the serialized_named_tensors should be the full weights of a decoder layer or the misc weights
        (embedding, norm, lm_haed). You should set finished=True when you call this function for the last time.
        """

        def _construct(item):
            """Deserialize torch.Tensor
            Args:
                item (Tuple[Callable, Tuple]): the return of reduce_tensor
            """
            func, args = item
            args = list(args)
            args[6] = torch.cuda.current_device()  # device id.
            return func(*args).clone()

        if not hasattr(self, "_export_iter"):
            self._get_model_params()
            que = Queue()
            tm_model = self._tm_model
            tm_model.input_model.model_path = que
            self._update_params_que = que
            self._export_iter = tm_model.export_iter()

        with torch.cuda.device(self.devices[0]):
            if isinstance(request.serialized_named_tensors, str):
                weights = ForkingPickler.loads(
                    base64.b64decode(request.serialized_named_tensors)
                )
                weights = {k: _construct(v) for k, v in weights}
            else:
                weights = request.serialized_named_tensors
            self._update_params_que.put(weights)
            next(self._export_iter)

        if request.finished:
            self._check_unloaded_tm_params()
            self._process_weights()
            if self._engine_created is False:
                self._create_engine()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        model_name: str = None,
        chat_template_name: str = None,
        engine_config: TurbomindEngineConfig = None,
        **kwargs,
    ):
        """LMDeploy's turbomind inference engine.

        Args:
            pretrained_model_name_or_path (str):
                It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                      converted by `lmdeploy convert` command or download from
                      ii) and iii)
                    - ii) The model_id of a lmdeploy-quantized model hosted
                      inside a model repo on huggingface.co, such as
                      "InternLM/internlm-chat-20b-4bit",
                      "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                      on huggingface.co, such as "internlm/internlm-chat-7b",
                      "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                      and so on.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update configuration when initialize the engine.
        """
        return cls(
            model_path=pretrained_model_name_or_path,
            model_name=model_name,
            chat_template_name=chat_template_name,
            engine_config=engine_config,
            **kwargs,
        )

    def close(self):
        if hasattr(self, "_tm_model"):
            # close immediately after init engine with empty_init=True
            self._tm_model.tm_params.clear()
        if hasattr(self, "_export_iter"):
            del self._export_iter
        if self.model_comm is not None:
            self.model_comm = None
        self._engine_created = False

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind or drift instance."""
        if getattr(self, "_engine_type", "turbomind") == "drift":
            self._ensure_drift_engine()
            return TurboMindInstance(self, self.config, cuda_stream_id)
        return TurboMindInstance(self, self.config, cuda_stream_id)


    def get_schedule_metrics(self):
        # TODO: support dp
        tm_metrics = self.model_comm.get_schedule_metrics(0, 0)
        return ScheduleMetrics(
            active_seqs=tm_metrics.active_seqs,
            waiting_seqs=tm_metrics.waiting_seqs,
            total_blocks=tm_metrics.total_blocks,
            active_blocks=tm_metrics.active_blocks,
            cached_blocks=tm_metrics.cached_blocks,
            free_blocks=tm_metrics.free_blocks,
        )


def _get_logits(outputs, offset: int):
    logits = outputs["logits"]

    def _func(out: EngineOutput, step: int, **kwargs):
        out.logits = logits[: step - offset - 1, :]

    return _func


def _get_last_hidden_state(outputs, offset: int):
    last_hidden_state = outputs["last_hidden_state"]

    def _func(out: EngineOutput, step: int, **kwargs):
        out.last_hidden_state = last_hidden_state[: step - offset - 1, :]

    return _func


def _get_logprobs_impl(
    logprob_vals: torch.Tensor,
    logprob_idxs: torch.Tensor,
    logprob_nums: torch.Tensor,
    output_ids: List[int],
    logprobs: int,
    offset: int,
):
    """Get logprob of each generated token.

    Args:
        logprob_vals (torch.Tensor): shape (max_new_tokens, 1024),
            1024 is the max_logprobs that turbomind engine can output
        logprob_idxs (torch.Tensor): shape (max_new_tokens, 1024)
        logprob_nums (torch.Tensor): shape (max_new_tokens,)
        output_ids (List[int]): new generated token ids
        logprobs (int): top n logprobs to return
        offset (int): offset to index logprob_vals, logprob_idxs and logprob_nums.
            It indicates where to start getting logprobs for the current generated tokens `output_ids`
    """
    out_logprobs = []
    # the total generated token number until now
    length = len(output_ids) + offset
    for pos, idx, val, n in zip(
        range(len(output_ids)),
        logprob_idxs[offset:length],
        logprob_vals[offset:length],
        logprob_nums[offset:length],
    ):
        topn = min(n.item(), logprobs)
        tok_res = {idx[i].item(): val[i].item() for i in range(topn)}
        token_id = output_ids[pos]
        if token_id not in tok_res:
            valid_n = n.item()
            tok_res[token_id] = val[:valid_n][idx[:valid_n] == token_id].item()
        ids = list(tok_res.keys())
        for k in ids:
            if tok_res[k] == float("-inf"):
                tok_res.pop(k)
        out_logprobs.append(tok_res)
    return out_logprobs


def _get_logprobs(outputs, output_logprobs: int):
    logprob_vals = outputs["logprob_vals"]  # shape {max_new_tokens, 1024}
    logprob_idxs = outputs["logprob_indexes"]  # shape {max_new_tokens, 1024}
    logprob_nums = outputs["logprob_nums"]  # shape {max_new_tokens,}
    offset = 0  # offset to index logprob_vals, logprob_idxs and logprob_nums

    def _func(out: EngineOutput, step: int, **kwargs):
        nonlocal offset
        out.logprobs = _get_logprobs_impl(
            logprob_vals,
            logprob_idxs,
            logprob_nums,
            out.token_ids,
            output_logprobs,
            offset,
        )
        offset += len(out.token_ids)

    return _func


def _get_metrics(metrics, eagle_metrics_debug: bool = False):
    import time

    from lmdeploy.messages import EngineEvent, EventType, RequestMetrics

    is_first = True

    def _func(out: EngineOutput, step: int, **kwargs):
        nonlocal is_first

        def _field(name: str, default: int | float = 0):
            return getattr(metrics, name, default)

        if not is_first:
            out.req_metrics = RequestMetrics(token_timestamp=time.time())
        else:
            events = [
                EngineEvent(EventType.QUEUED, metrics.enque_time / 1000000),
                EngineEvent(
                    EventType.SCHEDULED, metrics.scheduled_time / 1000000
                ),
            ]
            out.req_metrics = RequestMetrics(
                token_timestamp=time.time(), engine_events=events
            )
            is_first = False

        # Attach TurboMind EAGLE speculative decoding stats when available.
        eagle_steps = _field("eagle_steps", 0)
        if eagle_steps > 0:
            num_draft = _field("eagle_total_draft_tokens", 0)
            num_accept = _field("eagle_total_accepted_tokens", 0)
            avg_accepted_per_step = (
                float(num_accept) / float(eagle_steps) if eagle_steps > 0 else 0.0
            )
            spec_info = {
                "num_drafts": int(eagle_steps),
                "num_draft_tokens": int(num_draft),
                "num_accepted_tokens": int(num_accept),
                "avg_accepted_per_step": avg_accepted_per_step,
            }

            # Extended per-request EAGLE metrics used by SpeculativeDecodingStats
            # / EagleMetricsSummary. These are interpreted as per-step averages
            # over the speculative run.
            tokens_per_seq = 0.0
            accepted_len = 0.0
            max_accepted_len = 0
            committed_extras = 0.0

            if eagle_steps > 0 and num_draft > 0:
                # Approximate tokens drafted per speculative step. For the
                # common single-batch case this equals the planned
                # tokens_per_seq used by the engine.
                tokens_per_seq = float(num_draft) / float(eagle_steps)

            if eagle_steps > 0:
                # Average accepted length per speculative step (without the
                # "+1 bonus" convention; Python applies that when building
                # summaries). In v1 we expose this as both accepted_len and
                # max_accepted_len, and treat committed_extras as
                # accepted_len - 1 clamped at zero.
                accepted_len = avg_accepted_per_step
                max_accepted_len = int(accepted_len)
                committed_extras = max(0.0, accepted_len - 1.0)

            spec_info.update(
                tokens_per_seq=float(tokens_per_seq),
                accepted_len=float(accepted_len),
                max_accepted_len=int(max_accepted_len),
                committed_extras=float(committed_extras),
            )

            # Optional target-tree decode metrics. These are only
            # populated by the TurboMind backend when the engine
            # enables the EAGLE3 target-tree path; otherwise they
            # remain zero and we omit the nested block.
            tree_draft = _field("eagle_tree_draft_tokens", 0)
            tree_target = _field("eagle_tree_target_tokens", 0)
            tree_accepted = _field("eagle_tree_accepted_tokens", 0)
            if tree_draft or tree_target or tree_accepted:
                spec_info["tree_decode"] = {
                    "num_tree_draft_tokens": int(tree_draft),
                    "num_tree_target_tokens": int(tree_target),
                    "num_tree_accepted_tokens": int(tree_accepted),
                }

            # KV rewind metrics (optional; default to zero until wired in C++).
            total_rewound = _field("eagle_total_rewound_tokens", 0)
            rewind_steps = _field("eagle_rewind_steps", 0)
            if total_rewound or rewind_steps:
                spec_info["num_rewound_tokens"] = int(total_rewound)
                spec_info["rewind_steps"] = int(rewind_steps)

            # Additional EAGLE step-level aggregates from the backend.
            max_tokens_per_seq = _field("eagle_max_tokens_per_seq", 0)
            max_accepted_len_backend = _field("eagle_max_accepted_len", 0)
            steps_accept_ge2 = _field("eagle_steps_accept_ge2", 0)
            total_committed_extras = _field("eagle_total_committed_extras", 0)
            if max_tokens_per_seq:
                spec_info["max_tokens_per_seq"] = int(max_tokens_per_seq)
            if max_accepted_len_backend:
                spec_info["max_accepted_len"] = int(max_accepted_len_backend)
            if steps_accept_ge2:
                spec_info["steps_accept_ge2"] = int(steps_accept_ge2)
            if total_committed_extras:
                spec_info["total_committed_extras"] = int(total_committed_extras)
            out.req_metrics.spec_info = spec_info

            # Optional debug/trace logging for EAGLE metrics.
            if eagle_metrics_debug:
                logger.info(
                    "[EAGLE][Metrics] steps=%d num_draft_tokens=%d num_accepted_tokens=%d "
                    "avg_accepted_per_step=%.4f num_rewound_tokens=%d rewind_steps=%d",
                    int(eagle_steps),
                    int(num_draft),
                    int(num_accept),
                    float(avg_accepted_per_step),
                    int(total_rewound),
                    int(rewind_steps),
                )
        else:
            # Explicitly mark EAGLE as disabled for this request.
            out.req_metrics.spec_info = None

    return _func


class StreamingSemaphore:

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.fut = None
        self.val = 0

    async def acquire(self):
        if self.val:
            self.val = 0
            return
        self.fut = self.loop.create_future()
        await self.fut
        self.fut = None
        self.val = 0

    def release(self):
        if not self.val:
            self.val = 1
            if self.fut:
                self.fut.set_result(None)


class TurboMindInstance:
    """Instance of TurboMind.

    Args:
        tm_model (str): turbomind's model path
        cuda_stream_id(int): identity of a cuda stream
    """

    def __init__(
        self, tm_model: TurboMind, config: TurbomindModelConfig, cuda_stream_id: int = 0
    ):
        self.tm_model = tm_model
        self.cuda_stream_id = cuda_stream_id

        self.node_id = tm_model.node_id
        self.gpu_count = tm_model.gpu_count

        self.session_len = tm_model.session_len

        # Drift path: reuse shared DriftEngine wrapper as the model instance
        self.errcode_map = {
            0: ResponseType.SUCCESS,
            1: ResponseType.SESSION_NOT_EXIST,
            2: ResponseType.SESSION_REPEAT,
            3: ResponseType.SESSION_REPEAT,
            4: ResponseType.INTERNAL_ENGINE_ERROR,
            5: ResponseType.INTERNAL_ENGINE_ERROR,
            6: ResponseType.INPUT_LENGTH_ERROR,
            7: ResponseType.FINISH,
            8: ResponseType.CANCEL,
            9: ResponseType.PREFIX_CACHE_CONFLICT_INTERACTIVE_MODE,
            -1: ResponseType.INTERNAL_ENGINE_ERROR,
        }

        if getattr(self.tm_model, "_engine_type", "turbomind") == "drift":
            self._model_inst = self.tm_model._drift_engine
            self.config = config
            self.lock = None
            return

        # create model instances
        lazy_init = self.tm_model.config_dict["engine_config"].get("empty_init", False)
        self._model_inst = None if lazy_init else self._create_model_instance(0)

        self.config = config
        self.lock = None
        # error code map from csrc (refer to `struct Request` in src/turbomind/engine/request.h)
        # to lmdeploy.messages.ResponseType


    @property
    def model_inst(self):
        if self._model_inst is None:
            self._model_inst = self._create_model_instance(0)
        return self._model_inst

    def _create_model_instance(self, device_id):
        model_inst = self.tm_model.model_comm.create_model_instance(device_id)
        return model_inst

    def _get_extra_output_processors(
        self,
        outputs: Dict[str, torch.Tensor],
        gen_config: GenerationConfig,
        input_len: int,
        metrics: "_tm.RequestMetrics",
    ):

        def _get_offset(type):
            return input_len - 1 if type == "generation" else 0

        fs = []
        if gen_config.output_logits:
            offset = _get_offset(gen_config.output_logits)
            fs.append(_get_logits(outputs, offset))
        if gen_config.output_last_hidden_state:
            offset = _get_offset(gen_config.output_last_hidden_state)
            fs.append(_get_last_hidden_state(outputs, offset))
        if gen_config.logprobs:
            fs.append(_get_logprobs(outputs, gen_config.logprobs))
        if self.tm_model.engine_config.enable_metrics:
            spec_cfg = getattr(self.tm_model.engine_config, "speculative_config", None)
            eagle_metrics_debug = bool(
                getattr(spec_cfg, "eagle_metrics_debug", False)
            ) if spec_cfg is not None else False
            fs.append(_get_metrics(metrics, eagle_metrics_debug))
        return fs

    def prepare_embeddings(self, input_embeddings=None, input_embedding_ranges=None):
        """Convert embeddings."""
        if input_embeddings is None:
            return None, None

        assert len(input_embeddings) == len(input_embedding_ranges)
        if not isinstance(input_embeddings[0], (list, type(None))):
            input_embeddings = [input_embeddings]
            input_embedding_ranges = [input_embedding_ranges]

        if all([isinstance(x, type(None)) for x in input_embeddings]):
            return None, None

        hidden_dim = None
        for embeddings in input_embeddings:
            if embeddings is not None:
                hidden_dim = embeddings[0].squeeze().shape[-1]
                break
        assert hidden_dim is not None

        # construct input_embeddings
        for i in range(len(input_embeddings)):
            item = input_embeddings[i] or []
            # convert to torch.Tensor if input is np.ndarray
            if item and isinstance(item[0], np.ndarray):
                item = [torch.from_numpy(x).squeeze() for x in item]
            # convert to lookup table type
            _MAP = dict(
                float=torch.float,
                bfloat16=torch.bfloat16,
                float16=torch.float16,
                fp8=torch.bfloat16,
            )
            dtype = _MAP.get(self.tm_model.config.weight_type, torch.float16)
            item = [x.to(dtype=dtype) for x in item]
            item = item or [torch.zeros(0, hidden_dim, dtype=dtype)]
            input_embeddings[i] = item
        input_embeddings = [torch.cat(x) for x in input_embeddings]
        input_embeddings = pad_sequence(input_embeddings, batch_first=True)
        input_embeddings = input_embeddings.reshape(input_embeddings.shape[0], -1).view(
            torch.int8
        )
        # construct input_embedding_ranges
        for i in range(len(input_embedding_ranges)):
            item = input_embedding_ranges[i] or []
            item = torch.IntTensor(item).reshape(-1, 2)
            input_embedding_ranges[i] = item
        input_embedding_ranges = pad_sequence(
            input_embedding_ranges, batch_first=True, padding_value=-1
        )

        return input_embeddings, input_embedding_ranges

    def prepare_mrope(self, input_meta: Dict[str, Any], input_len: int):
        mrope_position_ids = input_meta["mrope_position_ids"]
        mrope_position_delta = input_meta["mrope_position_delta"]
        assert mrope_position_ids.size(-1) == input_len
        mrope_position_ids = mrope_position_ids.t().contiguous()
        return mrope_position_ids, mrope_position_delta

    def prepare_inputs(
        self,
        input_ids,
        gen_config: GenerationConfig,
        input_embeddings=None,
        input_embedding_ranges=None,
        input_meta: Dict[str, Any] = None,
    ):
        """Convert inputs format."""
        assert isinstance(input_ids, Sequence)

        input_ids = torch.IntTensor(input_ids)
        input_len = len(input_ids)

        inputs = dict(
            input_ids=input_ids,
        )

        input_embeddings, input_embedding_ranges = self.prepare_embeddings(
            input_embeddings, input_embedding_ranges
        )
        if input_embeddings is not None:
            inputs["input_embeddings"] = input_embeddings.cpu()
            inputs["input_embedding_ranges"] = input_embedding_ranges

        if input_meta and "mrope_position_ids" in input_meta:
            mrope_position_ids, mrope_position_delta = self.prepare_mrope(
                input_meta, input_len
            )
            inputs["mrope_position_ids"] = mrope_position_ids.type(torch.int32)
            inputs["mrope_position_delta"] = mrope_position_delta.type(torch.int32)
            inputs["mrope_length"] = torch.IntTensor([mrope_position_ids.shape[0]])

        return inputs, input_len

    async def async_cancel(self, session_id: int = None):
        self.model_inst.cancel()

    def async_end_cb(self, fut: asyncio.Future, status: int):
        """Executing on engine's signaling thread."""
        logger.info(f"[async_end_cb] session ended, status = {status}")
        fut.get_loop().call_soon_threadsafe(fut.set_result, status)

    async def async_end(self, session_id):
        fut = asyncio.get_running_loop().create_future()
        self.model_inst.end(partial(self.async_end_cb, fut), session_id)
        await fut

    def async_signal_cb(self, s: StreamingSemaphore):
        """Executing on engine's signaling thread."""
        s.loop.call_soon_threadsafe(s.release)

    async def async_stream_infer(
        self,
        session_id,
        input_ids,
        input_embeddings=None,
        input_embedding_ranges=None,
        input_meta: Dict[str, Any] = None,
        sequence_start: bool = True,
        sequence_end: bool = False,
        step=0,
        gen_config: GenerationConfig = None,
        stream_output=False,
        **kwargs,
    ):
        """Perform model inference.

        Args:
            session_id (int): the id of a session
            input_ids (numpy.ndarray): the token ids of a prompt
            input_embeddings (List[numpy.ndarray]): embeddings features
            input_embedding_ranges (List[Tuple[int,int]]): the begin/end
              offsets of input_embeddings to input_ids
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): indicator for cancelling the session
            gen_config (GenerationConfig): generation config
            stream_output (bool): indicator for stream output
            kwargs (dict): kwargs for backward compatibility
        """
        logger.info(f"[async_stream_infer] session {session_id} start")
        gen_cfg = self._get_generation_config(gen_config)

        inputs, input_len = self.prepare_inputs(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            input_meta=input_meta,
            gen_config=gen_config,
        )

        if gen_config.response_format is not None:
            tokenizer = self.tm_model.tokenizer
            vocab_size = self.tm_model.config.model_config.vocab_size

            try:
                tokenizer_info = TokenizerInfo.from_huggingface(
                    tokenizer.model.model, vocab_size=vocab_size
                )
                decode_grammar_type = gen_config.response_format["type"]
                if decode_grammar_type == "json_schema":
                    decode_grammar = gen_config.response_format[decode_grammar_type][
                        "schema"
                    ]
                elif decode_grammar_type == "regex_schema":
                    decode_grammar = gen_config.response_format[decode_grammar_type]
                elif decode_grammar_type == "json_object":
                    decode_grammar = '{"type" : "object", "additionalProperties": true}'

                compiler = _xgr.GrammarCompiler(tokenizer_info)

                if decode_grammar_type == "json_schema":
                    decode_grammar = json.dumps(decode_grammar)
                    grammar = compiler.compile_json_schema(decode_grammar)
                elif decode_grammar_type == "regex_schema":
                    decode_grammar = str(decode_grammar)
                    grammar = compiler.compile_regex(decode_grammar)
                elif decode_grammar_type == "json_object":
                    decode_grammar = str(decode_grammar)
                    grammar = compiler.compile_json_schema(decode_grammar)
                else:
                    assert False, (
                        f"Decode grammar type {decode_grammar_type} should be in "
                        '["json_schema", "regex_schema", "json_object"]'
                    )

                self.model_inst.set_grammar(grammar)
            except ValueError as e:
                logger.warning(
                    f"Failed to initialize guided decoding for tokenizer {tokenizer}, "
                    f"disable guided decoding: {e}"
                )
                gen_config.response_format = None

        session = _tm.SessionParam(
            id=session_id, step=step, start=sequence_start, end=sequence_end
        )

        inputs = _np_dict_to_tm_dict(inputs)

        sem = StreamingSemaphore()
        signal_cb = partial(self.async_signal_cb, sem)

        outputs, shared_state, metrics = self.model_inst.forward(
            inputs,
            session,
            gen_cfg,
            stream_output,
            self.tm_model.engine_config.enable_metrics,
            signal_cb,
        )

        outputs = _tm_dict_to_torch_dict(outputs)

        extra_fs = self._get_extra_output_processors(
            outputs, gen_config, input_len, metrics
        )

        output_ids_buf = outputs["output_ids"]

        finish = False
        state = None

        output_ids = []
        prev_len = step + input_len
        try:
            while True:
                await sem.acquire()
                state = shared_state.consume()

                status, seq_len = state.status, state.seq_len
                ret_status = ResponseType.SUCCESS

                if status in [7, 8]:  # finish / canceled
                    finish = True
                    ret_status = (
                        ResponseType.FINISH if status == 7 else ResponseType.CANCEL
                    )
                elif status:
                    logger.error(f"internal error. status_code {status}")
                    yield self._get_error_output(status)
                    break

                if seq_len == prev_len and not finish:
                    continue

                output_ids = output_ids_buf[prev_len:seq_len].tolist()
                output = EngineOutput(ret_status, output_ids)

                for f in extra_fs:
                    f(output, seq_len)

                prev_len = seq_len

                yield output

                if finish:
                    break

        except (GeneratorExit, asyncio.CancelledError) as e:
            logger.info(f"[async_stream_infer] {type(e).__name__}")
            self.model_inst.cancel()
        except Exception as e:
            logger.error(f"[async_stream_infer] {type(e).__name__} {e}")
            self.model_inst.cancel()
            yield self._get_error_output(-1)
        finally:
            # Contract: `cb` won't be called again if status is non-zero
            # wait for status to be set as `finish` or `error`
            while not state or state.status == 0:
                await sem.acquire()
                state = shared_state.consume()
            logger.info(f"[async_stream_infer] session {session_id} done")

    def _get_error_output(self, status):
        return EngineOutput(status=self.errcode_map[status], token_ids=[])

    def _get_generation_config(self, cfg: GenerationConfig):
        c = _tm.GenerationConfig()
        perf_mode_env = os.getenv("LMDEPLOY_EAGLE_PERF_MODE", "").strip().lower()
        perf_mode = perf_mode_env in ("1", "true", "yes", "on")
        c.max_new_tokens = cfg.max_new_tokens
        c.top_k = cfg.top_k
        c.top_p = cfg.top_p
        c.min_p = cfg.min_p
        c.temperature = cfg.temperature
        if cfg.stop_token_ids:
            c.eos_ids = cfg.stop_token_ids
        if cfg.bad_token_ids:
            c.bad_ids = _construct_stop_or_bad_words(cfg.bad_token_ids)
        # In perf mode we want GPU tail to remain usable and avoid
        # configuring stop-words for the tail path. Keep EOS ids so
        # decode semantics are correct, but skip stop_ids entirely.
        if not perf_mode and (not cfg.ignore_eos and cfg.stop_token_ids):
            c.stop_ids = _construct_stop_or_bad_words(cfg.stop_token_ids)
        c.repetition_penalty = cfg.repetition_penalty
        if cfg.min_new_tokens:
            c.min_new_tokens = cfg.min_new_tokens
        output_type = dict(all=1, generation=2)
        if cfg.output_last_hidden_state:
            c.output_last_hidden_state = output_type[cfg.output_last_hidden_state]
        if cfg.output_logits:
            c.output_logits = output_type[cfg.output_logits]
        if cfg.logprobs:
            if cfg.logprobs > MAX_LOGPROBS:
                cfg.logprobs = MAX_LOGPROBS
                logger.warning(
                    f"logprobs shoudd be in range [1, {MAX_LOGPROBS}]"
                    f"update logprobs={cfg.logprobs}"
                )
            c.output_logprobs = cfg.logprobs
        if cfg.random_seed is not None:
            c.random_seed = cfg.random_seed
        # print (c)
        return c
