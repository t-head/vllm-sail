# ruff: noqa: F821, E731, I001, UP037, UP032
# SPDX-License-Identifier: Apache-2.0
"""PPU profiling-aware serve benchmark.

Run ``python -m vllm_sail.benchmarks.serve`` with upstream bench-serve options.
``--skip-first-concurrency`` starts SDK profiling at request 2*max_concurrency,
matching the fork. It does not exclude those requests from reported metrics.
"""

from __future__ import annotations
import logging
from vllm.benchmarks import serve as _serve
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_sail.patch.bodies import bind_body


def _start_active_profile():
    try:
        from model_prof import start_active_profile
    except ImportError:
        logging.getLogger(__name__).warning(
            "model_prof is unavailable; active profiling is skipped"
        )
        return
    start_active_profile()


async def benchmark(
    task_type: TaskType,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: TokenizerLike | None,
    input_requests: list[SampleRequest],
    logprobs: int | None,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: int | None,
    lora_modules: Iterable[str] | None,
    extra_headers: dict | None,
    extra_body: dict | None,
    lora_assignment: Literal["random", "round-robin"] = "random",
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    ready_check_timeout_sec: int = 600,
    ssl_context: ssl.SSLContext | bool | None = None,
    self_timed: bool = False,
    probe_request_rate: float = 0.0,
    # PPU MODIFICATION: begin
    skip_first_concurrency: bool = False,
    # PPU MODIFICATION: end
):
    # PPU MODIFICATION: begin
    from vllm_sail.benchmarks.serve import _start_active_profile as start_active_profile
    if skip_first_concurrency and (max_concurrency is None or max_concurrency <= 0):
        raise ValueError("--skip-first-concurrency requires positive --max-concurrency")
    # PPU MODIFICATION: end
    try:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    except KeyError:
        raise ValueError(f"Unknown backend: {endpoint_type}") from None

    # Reuses connections across requests to reduce TLS handshake overhead.
    # Use ssl_context if provided, otherwise default to True for https URLs
    ssl_setting = ssl_context if ssl_context is not None else ("https://" in api_url)
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=ssl_setting,
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )
    test_extra_body = _merge_overrides(extra_body, input_requests[0].request_overrides)
    test_chat_messages = input_requests[0].chat_messages

    assert (
        test_mm_content is None
        or isinstance(test_mm_content, dict)
        or (
            isinstance(test_mm_content, list)
            and all(isinstance(item, dict) for item in test_mm_content)
        )
    ), "multi_modal_data must be a dict or list[dict]"
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_headers=extra_headers,
        extra_body=test_extra_body,
        chat_messages=test_chat_messages,
    )

    if ready_check_timeout_sec > 0:
        test_output = await wait_for_endpoint(
            request_func,
            test_input,
            session,
            timeout_seconds=ready_check_timeout_sec,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark "
                "arguments are correctly specified. "
                f"Error: {test_output.error}"
            )
        else:
            print("Initial test run completed.")
    else:
        print("Skipping endpoint ready check.")

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )
        warmup_tasks = []

        async def warmup_limited_request_func():
            async with warmup_semaphore:
                return await request_func(
                    request_func_input=test_input, session=session, pbar=warmup_pbar
                )

        for _ in range(num_warmups):
            request_task = asyncio.create_task(warmup_limited_request_func())
            warmup_tasks.append(request_task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        print("Warmup run completed.")

    print("Starting main benchmark run...")

    lora_modules_iter: Iterator[str] | None = None
    if lora_modules:
        lora_modules_list = list(lora_modules)
        if lora_assignment == "round-robin":
            # Deterministic round-robin assignment across requests.
            lora_modules_iter = iter(
                [
                    lora_modules_list[i % len(lora_modules_list)]
                    for i in range(len(input_requests))
                ]
            )
        else:
            # For each input request, choose a LoRA module at random.
            lora_modules_iter = iter(
                [random.choice(lora_modules_list) for _ in range(len(input_requests))]
            )

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=test_extra_body,
            chat_messages=test_chat_messages,
        )
        profile_output = await request_func(
            request_func_input=profile_input, session=session
        )
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    if not self_timed:
        if ramp_up_strategy is not None:
            print(f"Traffic ramp-up strategy: {ramp_up_strategy}.")
            print(
                f"Will increase RPS from {ramp_up_start_rps} to "
                f"{ramp_up_end_rps} RPS over the duration of the benchmark."
            )
        else:
            print(f"Traffic request rate: {request_rate}")

        print(f"Burstiness factor: {burstiness} ({distribution})")
        print(f"Maximum request concurrency: {max_concurrency}")
    else:
        print("Self timing is set, using the timestamps from the trace file.")

    spec_decode_metrics_before = await fetch_spec_decode_metrics(base_url, session)
    diffusion_metrics_before = await fetch_diffusion_metrics(base_url, session)

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    # PPU MODIFICATION: begin
    async def limited_request_func(request_func_input, session, pbar, task_id):
    # PPU MODIFICATION: end
        async with semaphore:
            # PPU MODIFICATION: begin
            if skip_first_concurrency:
                if task_id == 2 * max_concurrency:
                    print(f"Start active profiling with request id: {task_id}")
                    start_active_profile()
            elif task_id == 1:
                print("Start active profiling")
                start_active_profile()
            # PPU MODIFICATION: end
            return await request_func(
                request_func_input=request_func_input, session=session, pbar=pbar
            )

    probe_outputs: list[RequestFuncOutput] = []
    probe_stop = asyncio.Event()

    async def probe_loop():
        probe_input = replace(
            test_input,
            prompt="Hi",
            prompt_len=1,
            output_len=1,
            multi_modal_content=None,
            chat_messages=None,
        )
        interval = 1 / probe_request_rate
        while not probe_stop.is_set():
            probe_outputs.append(
                await request_func(request_func_input=probe_input, session=session)
            )
            await asyncio.sleep(interval)

    probe_task: asyncio.Task | None = None
    if probe_request_rate > 0:
        print(f"Probe request rate: {probe_request_rate} req/s")
        probe_task = asyncio.create_task(probe_loop())

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    # PPU MODIFICATION: begin
    task_id = 0
    # PPU MODIFICATION: end

    rps_change_events = []
    last_int_rps = -1
    if ramp_up_strategy is not None and ramp_up_start_rps is not None:
        last_int_rps = ramp_up_start_rps
        rps_change_events.append(
            {
                "rps": last_int_rps,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async for request, current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
        self_timed,
    ):
        if ramp_up_strategy is not None:
            current_int_rps = int(current_request_rate)
            if current_int_rps > last_int_rps:
                timestamp = datetime.now().isoformat()
                for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                    rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                last_int_rps = current_int_rps
        prompt, prompt_len, output_len, mm_content, request_id = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
            request.request_id,
        )
        per_request_extra_body = _merge_overrides(extra_body, request.request_overrides)
        req_model_id, req_model_name = model_id, model_name
        if lora_modules_iter:
            req_lora_module = next(lora_modules_iter)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        mm_content_typed: dict[str, Any] | list[dict[str, Any]] | None = None
        if isinstance(mm_content, (dict, list)):
            mm_content_typed = mm_content

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len or 0,
            logprobs=logprobs,
            multi_modal_content=mm_content_typed,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=per_request_extra_body,
            request_id=request_id,
            chat_messages=request.chat_messages,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(
                    # PPU MODIFICATION: begin
                    request_func_input=request_func_input,
                    session=session,
                    pbar=pbar,
                    task_id=task_id,
                    # PPU MODIFICATION: end
                )
            )
        )
        # PPU MODIFICATION: begin
        task_id += 1
        # PPU MODIFICATION: end
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if probe_task is not None:
        probe_stop.set()
        await probe_task

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    spec_decode_metrics_after = await fetch_spec_decode_metrics(base_url, session)
    spec_decode_stats: dict[str, Any] | None = None
    if spec_decode_metrics_before is not None and spec_decode_metrics_after is not None:
        delta_drafts = (
            spec_decode_metrics_after.num_drafts - spec_decode_metrics_before.num_drafts
        )
        delta_draft_tokens = (
            spec_decode_metrics_after.num_draft_tokens
            - spec_decode_metrics_before.num_draft_tokens
        )
        delta_accepted = (
            spec_decode_metrics_after.num_accepted_tokens
            - spec_decode_metrics_before.num_accepted_tokens
        )
        per_pos_rates: list[float] = []
        if delta_drafts > 0:
            positions = sorted(
                set(spec_decode_metrics_before.accepted_per_pos.keys())
                | set(spec_decode_metrics_after.accepted_per_pos.keys())
            )
            for pos in positions:
                before_val = spec_decode_metrics_before.accepted_per_pos.get(pos, 0)
                after_val = spec_decode_metrics_after.accepted_per_pos.get(
                    pos, before_val
                )
                delta_pos = after_val - before_val
                per_pos_rates.append(delta_pos / delta_drafts)

        if delta_draft_tokens > 0:
            acceptance_rate = (delta_accepted / delta_draft_tokens) * 100
            acceptance_length = (
                1 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0
            )
            spec_decode_stats = {
                "num_drafts": delta_drafts,
                "draft_tokens": delta_draft_tokens,
                "accepted_tokens": delta_accepted,
                "acceptance_rate": acceptance_rate,
                "acceptance_length": acceptance_length,
                "per_position_acceptance_rates": per_pos_rates,
            }

    diffusion_metrics_after = await fetch_diffusion_metrics(base_url, session)
    diffusion_stats: dict[str, Any] | None = None
    if diffusion_metrics_before is not None and diffusion_metrics_after is not None:
        delta_steps = (
            diffusion_metrics_after.num_denoising_steps
            - diffusion_metrics_before.num_denoising_steps
        )
        delta_positions = (
            diffusion_metrics_after.num_canvas_positions
            - diffusion_metrics_before.num_canvas_positions
        )
        delta_committed = (
            diffusion_metrics_after.num_committed_tokens
            - diffusion_metrics_before.num_committed_tokens
        )
        if delta_steps > 0 and delta_committed > 0:
            block_size = delta_positions / delta_steps  # canvas length (CL)
            num_canvases = delta_committed / block_size  # = number of commit steps
            denoising_steps = delta_steps - num_canvases  # exclude commit steps
            diffusion_stats = {
                "denoising_steps": denoising_steps,
                "canvas_positions": delta_positions,
                "committed_tokens": delta_committed,
                "committed_throughput": delta_committed / benchmark_duration,
                "steps_per_canvas": denoising_steps / num_canvases,
                "committed_per_step": delta_committed / denoising_steps,
            }

    metrics: BenchmarkMetrics | EmbedBenchmarkMetrics
    actual_output_lens: list[int] | int
    if task_type == TaskType.GENERATION:
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
        )
    else:
        metrics = calculate_metrics_for_embeddings(
            outputs=outputs,
            dur_s=benchmark_duration,
            selected_percentiles=selected_percentiles,
        )
        actual_output_lens = 0

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    if isinstance(metrics, BenchmarkMetrics) and tokenizer:
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict and isinstance(metrics, BenchmarkMetrics):
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    if isinstance(metrics, BenchmarkMetrics):
        if tokenizer:
            print(
                "{:<40} {:<10.2f}".format(
                    "Output token throughput (tok/s):", metrics.output_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak output token throughput (tok/s):",
                    metrics.max_output_tokens_per_s,
                )
            )
        print(
            "{:<40} {:<10.2f}".format(
                "Peak concurrent requests:", metrics.max_concurrent_requests
            )
        )
        if metrics.rtfx > 0.0:
            print(
                "{:<40} {:<10.2f}".format(
                    "RTFx (Inverse Real-Time Factor):", metrics.rtfx
                )
            )
    if tokenizer:
        print(
            "{:<40} {:<10.2f}".format(
                "Total token throughput (tok/s):", metrics.total_token_throughput
            )
        )

    probe_stats: dict[str, Any] | None = None
    if probe_task is not None:
        probe_lats = [o.latency for o in probe_outputs if o.success]
        if probe_lats:
            probe_stats = {
                "probe_completed": len(probe_lats),
                "probe_failed": len(probe_outputs) - len(probe_lats),
                "probe_median_e2el_ms": float(np.median(probe_lats)) * 1000,
                "probe_p99_e2el_ms": float(np.percentile(probe_lats, 99)) * 1000,
                "probe_max_e2el_ms": float(max(probe_lats)) * 1000,
            }
            print("{s:{c}^{n}}".format(s="Probe Requests", n=50, c="-"))
            print(
                "{:<40} {:<10}".format(
                    "Probe requests completed:", probe_stats["probe_completed"]
                )
            )
            print(
                "{:<40} {:<10}".format(
                    "Probe requests failed:", probe_stats["probe_failed"]
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median probe E2EL (ms):", probe_stats["probe_median_e2el_ms"]
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P99 probe E2EL (ms):", probe_stats["probe_p99_e2el_ms"]
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Max probe E2EL (ms):", probe_stats["probe_max_e2el_ms"]
                )
            )

    result: dict[str, Any]
    if isinstance(metrics, BenchmarkMetrics):
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "failed": metrics.failed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else None,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": actual_output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "start_times": [output.start_time for output in outputs],
            "generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
            "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
            "max_concurrent_requests": metrics.max_concurrent_requests,
            "rtfx": metrics.rtfx,
        }
    else:
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "request_throughput": metrics.request_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "errors": [output.error for output in outputs],
        }

    if probe_stats is not None:
        result.update(probe_stats)

    if rps_change_events:
        result["rps_change_events"] = rps_change_events

    if spec_decode_stats is not None:
        result["spec_decode_acceptance_rate"] = spec_decode_stats["acceptance_rate"]
        result["spec_decode_acceptance_length"] = spec_decode_stats["acceptance_length"]
        result["spec_decode_num_drafts"] = int(spec_decode_stats["num_drafts"])
        result["spec_decode_draft_tokens"] = int(spec_decode_stats["draft_tokens"])
        result["spec_decode_accepted_tokens"] = int(
            spec_decode_stats["accepted_tokens"]
        )
        result["spec_decode_per_position_acceptance_rates"] = spec_decode_stats.get(
            "per_position_acceptance_rates", []
        )

    if diffusion_stats is not None:
        result["diffusion_committed_throughput"] = diffusion_stats[
            "committed_throughput"
        ]
        result["diffusion_steps_per_canvas"] = diffusion_stats["steps_per_canvas"]
        result["diffusion_committed_per_step"] = diffusion_stats["committed_per_step"]
        result["diffusion_committed_tokens"] = int(diffusion_stats["committed_tokens"])
        result["diffusion_denoising_steps"] = int(diffusion_stats["denoising_steps"])
        result["diffusion_canvas_positions"] = int(diffusion_stats["canvas_positions"])

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    if task_type == TaskType.GENERATION and tokenizer:
        process_one_metric("ttft", "TTFT", "Time to First Token")
        process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    if diffusion_stats is not None:
        print("{s:{c}^{n}}".format(s="Diffusion Decoding", n=50, c="-"))
        for label, key, value_fmt in (
            ("Committed throughput (tok/s):", "committed_throughput", "{:<10.2f}"),
            ("Denoising steps per canvas:", "steps_per_canvas", "{:<10.2f}"),
            ("Committed per denoising step:", "committed_per_step", "{:<10.2f}"),
            ("Committed tokens:", "committed_tokens", "{:<10d}"),
            ("Denoising steps:", "denoising_steps", "{:<10d}"),
            ("Canvas positions evaluated:", "canvas_positions", "{:<10d}"),
        ):
            value = diffusion_stats[key]
            if value_fmt.endswith("d}"):
                value = int(value)
            print("{:<40} ".format(label) + value_fmt.format(value))

    if spec_decode_stats is not None and diffusion_stats is None:
        print("{s:{c}^{n}}".format(s="Speculative Decoding", n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                "Acceptance rate (%):", spec_decode_stats["acceptance_rate"]
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Acceptance length:", spec_decode_stats["acceptance_length"]
            )
        )
        print("{:<40} {:<10}".format("Drafts:", int(spec_decode_stats["num_drafts"])))
        print(
            "{:<40} {:<10}".format(
                "Draft tokens:", int(spec_decode_stats["draft_tokens"])
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Accepted tokens:", int(spec_decode_stats["accepted_tokens"])
            )
        )
        per_pos = spec_decode_stats.get("per_position_acceptance_rates", [])
        if per_pos:
            print("Per-position acceptance (%):")
            for i, rate in enumerate(per_pos):
                print("{:<40} {:<10.2f}".format(f"  Position {i}:", rate * 100))

    print("=" * 50)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(
            request_func_input=profile_input, session=session
        )
        if profile_output.success:
            print("Profiler stopped")

    await session.close()
    return result


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    _serve.add_cli_args(parser)
    parser.add_argument(
        "--skip-first-concurrency",
        action="store_true",
        help="Delay SDK profiling until request 2*max_concurrency; all requests remain in metrics.",
    )
    args = parser.parse_args()
    implementation = bind_body(benchmark, _serve)

    async def run_benchmark(*positional, **kwargs):
        return await implementation(
            *positional, skip_first_concurrency=args.skip_first_concurrency, **kwargs
        )

    # Scoped to this process-local CLI entry; the regular vllm bench path is intact.
    _serve.benchmark = run_benchmark
    _serve.main(args)


if __name__ == "__main__":
    main()
