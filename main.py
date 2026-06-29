from __future__ import annotations
    
import os
import sys
import ast
import copy
import math
import time
import json
import textwrap
import traceback
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from queue import Empty
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
LLM4AD_ROOT = PROJECT_ROOT / "LLM4AD"
for path in (PROJECT_ROOT, LLM4AD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import *
from tsp_template import *
from tsp_response import *
from tsp_evaluation import TSPEvaluation
from frame_prompt import *
from LLM4AD.llm4ad.base.code import TextFunctionProgramConverter
from LLM4AD.llm4ad.tools.llm.llm_api_https import HttpsApi
from LLM4AD.llm4ad.tools.profiler import ProfilerBase
from LLM4AD.llm4ad.method.eoh import EoH

use_exist = False


def _is_valid_outer_score(score: float | None) -> bool:
    return score is not None and not math.isinf(score) and score <= 0


@dataclass
class MethodStats:
    call_count: int = 0
    used_budget: int = 0
    best_score: float | None = None
    last_score: float | None = None
    total_gain: float = 0.0
    success_count: int = 0
    last_gain: float = 0.0
    gain_history: list[float] = field(default_factory=list)
    benefit_history: list[float] = field(default_factory=list)
    used_operators: list[str] = field(default_factory=list)
    total_sample_calls: int = 0


class InnerProfiler:
    def __init__(self, log_dir: str):
        self._log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._samples_dir = os.path.join(self._log_dir, "samples")
        os.makedirs(self._samples_dir, exist_ok=True)
        self._run_log_path = os.path.join(self._log_dir, "run_log.txt")
        self._schedule_path = os.path.join(self._log_dir, "method_schedule.json")
        self._best_path = os.path.join(self._log_dir, "method_best.json")
        self._sample_order_history: list[int] = []
        self._score_history: list[float] = []
        self._method_histories: dict[str, dict[str, list[float]]] = {}
        self._method_boundaries: dict[str, list[tuple[int, int]]] = {}
        self._best_score = float("-inf")
        self._global_sample_cursor = 0
        self._logger = logging.getLogger(f"inner_profiler_{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.FileHandler(self._run_log_path, mode="w", encoding="utf-8")
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
            self._logger.addHandler(handler)

    def record_baseline(self, frame: AlgorithmFrame):
        self._logger.info("Baseline score: %s", frame.score)
        self._logger.info("Methods to evolve: %s", frame.method_order())

    def register_method_samples(
        self,
        method_name: str,
        local_start_order: int,
        actual_sample_count: int,
        valid_sample_orders: list[int],
        valid_sample_scores: list[float],
        round_start_score: float | None = None,
    ) -> None:
        if actual_sample_count <= 0:
            return
        valid_map = {
            order: score for order, score in zip(valid_sample_orders, valid_sample_scores)
        }
        global_start = self._global_sample_cursor
        self._global_sample_cursor += actual_sample_count

        current_global_best = self._best_score
        for offset in range(1, actual_sample_count + 1):
            local_order = local_start_order + offset
            if local_order in valid_map:
                current_global_best = max(current_global_best, valid_map[local_order])
            if current_global_best > float("-inf"):
                self._sample_order_history.append(global_start + offset)
                self._score_history.append(current_global_best)
        self._best_score = current_global_best

        if not valid_map:
            return
        if method_name not in self._method_histories:
            self._method_histories[method_name] = {"x": [], "y": []}
            self._method_boundaries[method_name] = []

        method_history = self._method_histories[method_name]
        start_x = global_start + 1
        end_x = global_start + actual_sample_count
        self._method_boundaries[method_name].append((start_x, end_x))

        current_method_best = method_history["y"][-1] if method_history["y"] else float("-inf")
        if round_start_score is not None and not math.isinf(round_start_score):
            current_method_best = max(current_method_best, round_start_score)
        for offset in range(1, actual_sample_count + 1):
            local_order = local_start_order + offset
            if local_order in valid_map:
                current_method_best = max(current_method_best, valid_map[local_order])
            method_history["x"].append(global_start + offset)
            method_history["y"].append(current_method_best)

    def register_method_step(self, content: dict):
        self._append_json(self._schedule_path, content)
        self._logger.info("======================================================")
        self._logger.info("Dispatch order : %s", content["dispatch_order"])
        self._logger.info("Method         : %s", content["method_name"])
        self._logger.info("Budget         : %s", content["budget"])
        self._logger.info("Benefit mode   : %s", content["benefit_mode"])
        self._logger.info("Selected score : %s -> %s", content["score_before"], content["score_after"])
        self._logger.info("Gain           : %s", content["gain"])
        self._logger.info("Operators      : %s", content["used_operators"])
        self._logger.info("Actual samples : %s", content["actual_sample_count"])
        self._logger.info("Remaining      : %s", content["remaining_budget"])
        self._logger.info("======================================================")
        score_after = content["score_after"]
        if _is_valid_outer_score(score_after):
            if score_after > self._best_score:
                self._best_score = score_after
                self._append_json(self._best_path, content)

    def finish(self, frame: AlgorithmFrame, final_summary: dict):
        self._append_json(self._best_path, {"final_frame": frame.description(), **final_summary})
        self._plot_convergence_curve()

    def record_test_result(self, test_summary: dict):
        self._append_json(self._best_path, {"test_result": test_summary})
        self._logger.info("Test score    : %s", test_summary.get("test_score"))
        self._logger.info("Test instances: %s", test_summary.get("num_test_instances"))

    def _append_json(self, path: str, content: dict):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(content)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _plot_convergence_curve(self):
        try:
            import matplotlib.pyplot as plt

            if self._score_history:
                best_so_far = []
                current_best = float("-inf")
                for score in self._score_history:
                    current_best = max(current_best, score)
                    best_so_far.append(current_best)

                plt.figure(figsize=(8, 5))
                plt.plot(self._sample_order_history, best_so_far, marker="o")
                plt.xlabel("Consumed Inner Samples")
                plt.ylabel("Best Frame Score So Far")
                plt.title("Inner-Only Best Score Curve")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self._log_dir, "convergence_inner.png"), dpi=150)
                plt.close()

            for method_name, history in self._method_histories.items():
                if not history["x"]:
                    continue
                plt.figure(figsize=(8, 5))
                plt.plot(history["x"], history["y"], marker="o", label=method_name)
                for start_x, _ in self._method_boundaries.get(method_name, []):
                    plt.axvline(start_x, linestyle="--", color="gray", alpha=0.35)
                plt.xlabel(f"Valid Samples Generated By {method_name}")
                plt.ylabel(f"Best Sample Score For {method_name} So Far")
                plt.title(f"Method Sample-Level Convergence - {method_name}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self._log_dir, f"convergence_method_{method_name}.png"),
                    dpi=150,
                )
                plt.close()
        except Exception:
            traceback.print_exc()

def _frame_evaluation_worker(
    instance: list[tuple[np.ndarray, float]],
    algorithm_body: str,
    timeout_seconds: int,
    result_queue: mp.Queue,
    score_mode: str = "gap",
    num_workers: int | None = None,
) -> None:
    try:
        evaluation = TSPEvaluation(
            instance=instance,
            algorithm_str=algorithm_body,
            timeout_seconds=timeout_seconds,
            score_mode=score_mode,
            num_workers=num_workers,
        )
        result_queue.put({"score": evaluation.evaluate_program()})
    except Exception:
        result_queue.put({"error": traceback.format_exc()})


def _stop_process(process: mp.Process) -> None:
    process.join(timeout=1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join()


class FrameProfiler:
    def __init__(self, log_dir: str, record_sep: int = 200):
        self._num_samples = 0
        self._best_score = float("-inf")
        self._record_sep = record_sep
        self._log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._samples_dir = os.path.join(self._log_dir, "samples")
        os.makedirs(self._samples_dir, exist_ok=True)
        self._sample_order_history = []
        self._score_history = []
        self._logger = logging.getLogger(f"frame_profiler_{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._create_log_file()

    def register_frame(self, frame: AlgorithmFrame):
        # Keep the raw LLM reply in history logs so parsing issues can be traced later.
        self._num_samples += 1
        content = {
            "sample_order": self._num_samples,
            "frame_id": frame.frame_id,
            "methods_to_evolve": list(frame.method_args.keys()),
            "description": frame.description(),
            "score": frame.score,
            "operator": frame.operator,
            "body": frame.body,
            "response_text": frame.response_text,
            "sample_time": frame.sample_time,
            "evaluate_time": frame.evaluate_time,
        }
        self._record_verbose(content)
        self._write_json(content, record_type="history")
        if _is_valid_outer_score(frame.score) and frame.score > self._best_score:
            self._best_score = frame.score
            self._write_json(content, record_type="best")
        if _is_valid_outer_score(frame.score):
            self._sample_order_history.append(self._num_samples)
            self._score_history.append(frame.score)

    def _write_json(self, content: dict, *, record_type: str):
        if record_type == "history":
            lower_bound = ((self._num_samples - 1) // self._record_sep) * self._record_sep
            upper_bound = lower_bound + self._record_sep
            filename = f"samples_{lower_bound + 1}~{upper_bound}.json"
        else:
            filename = "samples_best.json"
        path = os.path.join(self._samples_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(content)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _create_log_file(self):
        log_path = os.path.join(self._log_dir, "run_log.txt")
        if not self._logger.handlers:
            handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
            self._logger.addHandler(handler)

    def _record_verbose(self, content: dict):
        self._logger.info("======================================================")
        self._logger.info(f"Sample order : {content['sample_order']}")
        self._logger.info(f"Frame ID     : {content['frame_id']}")
        self._logger.info(f"Methods      : {content['methods_to_evolve']}")
        self._logger.info(f"Operator     : {content['operator']}")
        self._logger.info(f"Score        : {content['score']}")
        self._logger.info(f"Sample time  : {content['sample_time']}")
        self._logger.info(f"Evaluate time: {content['evaluate_time']}")
        self._logger.info("Description:")
        self._logger.info(content["description"])
        self._logger.info("======================================================")

    def finish(self):
        self._plot_convergence_curve()

    def _plot_convergence_curve(self):
        if not self._score_history:
            return
        try:
            import matplotlib.pyplot as plt

            best_so_far = []
            current_best = float("-inf")
            for score in self._score_history:
                current_best = max(current_best, score)
                best_so_far.append(current_best)

            plt.figure(figsize=(8, 5))
            plt.plot(self._sample_order_history, best_so_far, marker="o")
            plt.xlabel("Sample Order")
            plt.ylabel("Best Score So Far")
            plt.title("FrameEoH Best Score Curve")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self._log_dir, "convergence_frame.png"), dpi=150)
            plt.close()
        except Exception:
            traceback.print_exc()

class Inner:
    def __init__(self,
                 llm: HttpsApi,
                 train_instance: list[tuple[np.ndarray, float]],
                 test_instance: list[tuple[np.ndarray, float]] | None,
                 algorithm_frame: AlgorithmFrame,
                 eoh_log_root: str,
                 max_sample_nums: int = 20,
                 pop_size: int = 5,
                 budget_mode: str = "average",
                 benefit_mode: str = "absolute_gain",
                 method_selection_mode: str = "greedy",
                 per_call_budget_cap: int = 50,
                 discount_factor: float = 0.8,
                 softmax_temperature: float = 1.0,
                 hybrid_weights: tuple[float, float, float] = (1.0, 0.5, 0.25),
                 recent_window: int = 3,
                 use_adj_prev_operator: bool = True,
                 use_adj_next_operator: bool = True,
                 use_context_prompt: bool = True,
                 eval_num_workers: int | None = None):
        self.llm = llm
        self.train_instance = train_instance
        self.test_instance = test_instance or []
        self.algorithm_frame = algorithm_frame
        self.eoh_log_root = eoh_log_root
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
        self.budget_mode = budget_mode
        self.benefit_mode = benefit_mode
        self.method_selection_mode = method_selection_mode
        self.per_call_budget_cap = per_call_budget_cap
        self.discount_factor = discount_factor
        self.softmax_temperature = softmax_temperature
        self.hybrid_weights = hybrid_weights
        self.recent_window = recent_window
        self.use_adj_prev_operator = use_adj_prev_operator
        self.use_adj_next_operator = use_adj_next_operator
        self.use_context_prompt = use_context_prompt
        self.eval_num_workers = eval_num_workers
        self.method_order = self.algorithm_frame.method_order()
        self.remaining_budget = max_sample_nums
        self.total_consumed_budget = 0
        self.total_consumed_samples = 0
        self.stats = {method: MethodStats() for method in self.method_order}
        self.dispatch_order = 0
        self.profiler = InnerProfiler(log_dir=os.path.join(self.eoh_log_root, self.algorithm_frame.frame_id or "inner_only"))
        self.method_eoh_cache: dict[str, EoH] = {}
        self.method_best_individuals: dict[str, object] = {}
        self._initialize_method_best_individuals()
    
    def replace(self, method_name, func):
        code = self.algorithm_frame.body
        tree = ast.parse(code)
        lines = code.splitlines()
        method_src = textwrap.indent(func.to_code_without_docstring().rstrip(), "    ").splitlines()

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Algorithm":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == func.name:
                        lines[item.lineno - 1:item.end_lineno] = method_src
                        self.algorithm_frame.body = "\n".join(lines) + "\n"
                        break

        self.algorithm_frame.method_introduction[method_name] = func.algorithm
        self.method_best_individuals[method_name] = copy.deepcopy(func)

    def _initialize_method_best_individuals(self) -> None:
        # Initialize every method's incumbent from the original response so
        # adj_prev/adj_next can use neighboring methods from the first round.
        for method_name in self.method_order:
            method_code = self.algorithm_frame.method_code(method_name)
            if not method_code:
                continue
            func = TextFunctionProgramConverter.text_to_function(method_code)
            if func is None:
                continue
            func.algorithm = self.algorithm_frame.method_usage(method_name) or ""
            func.score = None
            func.evaluate_time = 0.0
            func.sample_time = 0.0
            func.operator = "response"
            self.method_best_individuals[method_name] = copy.deepcopy(func)

    def _refresh_population_head_score(
        self,
        evolve_frame: EoH,
        incumbent_score: float | None,
    ) -> bool:
        if not evolve_frame._population.population:
            return False
        evolve_frame._population.population[0].score = incumbent_score
        evolve_frame._population.clear_pending_offspring()
        return True
    
    def _evaluate_frame(self) -> float | None:
        return self._evaluate_body(self.algorithm_frame.body, self.train_instance, "absolute_distance")

    def _evaluate_test_frame(self) -> float | None:
        if not self.test_instance:
            return None
        return self._evaluate_body(self.algorithm_frame.body, self.test_instance, "gap")

    def _evaluate_body(
        self,
        algorithm_body: str,
        instance: list[tuple[np.ndarray, float]],
        score_mode: str,
    ) -> float | None:
        result_queue = mp.Queue()
        process = mp.Process(
            target=_frame_evaluation_worker,
            args=(instance, algorithm_body, 120, result_queue, score_mode, self.eval_num_workers),
        )
        process.start()
        try:
            result = result_queue.get(timeout=120)
        except Empty:
            return None
        finally:
            _stop_process(process)
            result_queue.close()
            result_queue.join_thread()
        if "error" in result:
            return None
        score = result["score"]
        if score is None or math.isinf(score):
            return None
        return score

    def _neighbor_context(self, method_name: str) -> dict:
        idx = self.method_order.index(method_name)
        prev_name = self.method_order[idx - 1] if idx > 0 else None
        next_name = self.method_order[idx + 1] if idx < len(self.method_order) - 1 else None
        return {
            "prev_name": prev_name,
            "next_name": next_name,
            "prev_method_individual": self.method_best_individuals.get(prev_name) if prev_name else None,
            "next_method_individual": self.method_best_individuals.get(next_name) if next_name else None,
        }

    def _calculate_gain(self, old_score: float | None, new_score: float | None) -> float:
        if old_score is None or new_score is None:
            return float("-inf")
        return new_score - old_score

    def _calculate_relative_gain(self, old_score: float | None, new_score: float | None) -> float:
        if old_score is None or new_score is None:
            return float("-inf")
        base = abs(old_score)
        if base == 0:
            return float("-inf")
        improvement = new_score - old_score
        if improvement < 0:
            return float("-inf")
        return abs(improvement) / base

    def _calculate_benefit_gain(self, old_score: float | None, new_score: float | None) -> float | None:
        if self.benefit_mode in {"relative_gain", "relative_discounted_gain"}:
            return self._calculate_relative_gain(old_score, new_score)
        return None

    def _recent_success_rate(self, method_name: str) -> float:
        gains = self.stats[method_name].gain_history[-self.recent_window:]
        if not gains:
            return 0.0
        return sum(1 for gain in gains if gain > 0) / len(gains)

    def _benefit_value(self, method_name: str) -> float:
        stats = self.stats[method_name]
        absolute_gain = stats.last_gain
        last_benefit_gain = stats.benefit_history[-1] if stats.benefit_history else float("-inf")
        if self.benefit_mode == "absolute_gain":
            return absolute_gain
        if self.benefit_mode == "relative_gain":
            return last_benefit_gain
        if self.benefit_mode == "relative_discounted_gain":
            if not stats.benefit_history:
                return float("-inf")
            total = 0.0
            for idx, gain in enumerate(reversed(stats.benefit_history)):
                total += gain * (self.discount_factor ** idx)
            return total
        w1, w2, w3 = self.hybrid_weights
        return w1 * absolute_gain + w2 * last_benefit_gain + w3 * self._recent_success_rate(method_name)

    def _budget_for_method(self, method_name: str, index: int = 0) -> int:
        if self.budget_mode == "average":
            base = self.max_sample_nums // max(1, len(self.method_order))
            remainder = self.max_sample_nums % max(1, len(self.method_order))
            return min(self.remaining_budget, base + (1 if index < remainder else 0))
        return min(self.remaining_budget, self.per_call_budget_cap)

    def _select_next_method(self) -> tuple[str | None, dict[str, float]]:
        benefit_map = {method: self._benefit_value(method) for method in self.method_order}
        candidate = max(benefit_map, key=benefit_map.get, default=None)
        if candidate is None:
            return None, benefit_map
        if self.method_selection_mode == "softmax":
            finite_items = [(method, value) for method, value in benefit_map.items() if not math.isinf(value)]
            if not finite_items:
                return candidate, benefit_map
            methods, values = zip(*finite_items)
            logits = np.array(values, dtype=float) / max(self.softmax_temperature, 1e-8)
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            return str(np.random.choice(methods, p=probs)), benefit_map
        return candidate, benefit_map

    def _update_method_stats(
        self,
        method: str,
        budget: int,
        score_after: float | None,
        gain: float,
        used_operators: list[str],
        benefit_gain: float | None = None,
    ) -> None:
        stats = self.stats[method]
        stats.call_count += 1
        stats.used_budget += budget
        stats.total_sample_calls += budget
        stats.last_score = score_after
        if score_after is not None:
            stats.best_score = score_after if stats.best_score is None else max(stats.best_score, score_after)
        stats.last_gain = gain
        stats.total_gain += 0 if gain == float("-inf") else gain
        stats.gain_history.append(gain)
        if gain > 0:
            stats.success_count += 1
        if benefit_gain is not None and not math.isinf(benefit_gain):
            stats.benefit_history.append(benefit_gain)
        stats.used_operators.extend(used_operators)

    def _record_method_step(
        self,
        method_name: str,
        budget: int,
        actual_sample_count: int,
        score_before: float | None,
        score_after: float | None,
        gain: float,
        used_operators: list[str],
        benefit_map: dict[str, float],
    ):
        self.dispatch_order += 1
        self.profiler.register_method_step(
            {
                "dispatch_order": self.dispatch_order,
                "method_name": method_name,
                "budget": budget,
                "actual_sample_count": actual_sample_count,
                "benefit_mode": self.benefit_mode,
                "benefit_values": benefit_map,
                "score_before": score_before,
                "score_after": score_after,
                "gain": gain,
                "used_operators": used_operators,
                "remaining_budget": self.remaining_budget,
                "consumed_budget_total": self.total_consumed_budget,
                "consumed_sample_total": self.total_consumed_samples,
            }
        )

    def _finalize_method_round(
        self,
        method: str,
        budget: int,
        actual_sample_count: int,
        score_before: float | None,
        evolve_frame: EoH,
        benefit_map: dict[str, float],
        best_func=None,
    ) -> None:
        self.remaining_budget -= budget
        self.total_consumed_budget += budget
        self.total_consumed_samples += actual_sample_count

        used_operators = getattr(evolve_frame, "_used_operator_history", [])
        score_after = self.algorithm_frame.score if best_func is None else best_func.score
        if best_func is not None:
            self.algorithm_frame.score = score_after
        gain = self._calculate_gain(score_before, score_after)
        benefit_gain = self._calculate_benefit_gain(score_before, score_after)

        self._update_method_stats(
            method,
            actual_sample_count,
            score_after,
            gain,
            used_operators,
            benefit_gain,
        )
        self._record_method_step(
            method,
            budget,
            actual_sample_count,
            score_before,
            score_after,
            gain,
            used_operators,
            benefit_map,
        )

    def _close_cached_eoh(self) -> None:
        for evolve_frame in self.method_eoh_cache.values():
            try:
                evolve_frame.close()
            except Exception:
                pass

    def _seed_method_population(self, evolve_frame: EoH, method_name: str) -> None:
        # Seed the current framework method into the initial population so the
        # first evolution round starts from the incumbent implementation.
        # The seed is treated as sample index 0 and does not consume budget.
        method_code = self.algorithm_frame.method_code(method_name)
        if not method_code:
            return
        func = TextFunctionProgramConverter.text_to_function(method_code)
        if func is None:
            return
        score = self.algorithm_frame.score
        if score is None or math.isinf(score):
            return
        func.score = score
        func.evaluate_time = 0.0
        func.sample_time = 0.0
        func.algorithm = self.algorithm_frame.method_usage(method_name) or ""
        func.operator = "i1"
        self.method_best_individuals[method_name] = copy.deepcopy(func)
        evolve_frame._used_operator_history.append("i1")
        evolve_frame._current_best_score = score
        evolve_frame._sample_order_history.append(0)
        evolve_frame._sample_score_history.append(score)
        evolve_frame._plot_sample_order_history.append(
            evolve_frame.get_plot_sample_order(0)
        )
        if not evolve_frame._population.has_duplicate_function(func):
            evolve_frame._population.population.append(func)
            evolve_frame._population.population.sort(key=lambda f: f.score, reverse=True)
        if evolve_frame._profiler is not None:
            evolve_frame._profiler.register_function_with_order(
                func,
                sample_order=0,
                program=self.algorithm_frame.body,
            )
            if hasattr(evolve_frame._profiler, "register_population"):
                evolve_frame._profiler.register_population(evolve_frame._population)

    def evolve_once(self, cur_method, cur_template_program, method_budget: int):
        cached_eoh = self.method_eoh_cache.get(cur_method)
        neighbor_context = self._neighbor_context(cur_method)
        history_start = None
        start_sample_count = None
        round_start_score = None
        if cached_eoh is None:
            frame_log_dir = os.path.join(
                self.eoh_log_root,
                self.algorithm_frame.frame_id or "unknown_frame",
                cur_method,
            )
            evaluation = TSPEvaluation(template_program=cur_template_program,
                                       task_description=task_description_inner,
                                       instance=self.train_instance, 
                                       method_name=cur_method,
                                       algorithm_str=self.algorithm_frame.body,
                                       score_mode="absolute_distance",
                                       num_workers=self.eval_num_workers)
            evolve_frame = EoH(llm=self.llm,
                               profiler=ProfilerBase(log_dir=frame_log_dir, log_style='complex'),
                               evaluation=evaluation,
                               max_sample_nums=method_budget,
                               max_generations=100,
                               pop_size=self.pop_size,
                               num_samplers=self.pop_size,
                               num_evaluators=self.pop_size,
                               use_e2_operator=True,
                               use_m1_operator=True,
                               use_m2_operator=True,
                               method_usage=self.algorithm_frame.method_usage(cur_method),
                               method_introduction=self.algorithm_frame.other_method_introductions(cur_method),
                               main_stream=self.algorithm_frame.main_stream(),
                               use_context_prompt=self.use_context_prompt,
                               use_adj_prev_operator=self.use_adj_prev_operator and neighbor_context["prev_name"] is not None,
                               use_adj_next_operator=self.use_adj_next_operator and neighbor_context["next_name"] is not None,
                                 **neighbor_context,
                                 keep_resources_alive=True,
                                 debug_mode=False)
            evolve_frame.set_plot_window(
                self.total_consumed_samples,
                0,
            )
            self._seed_method_population(evolve_frame, cur_method)
            # The seeded incumbent consumes one sample already, so keep the
            # original sample cap instead of granting an extra budget slot.
            evolve_frame._max_sample_nums = method_budget
            history_start = 0
            start_sample_count = 0
            round_start_score = self.algorithm_frame.score
        else:
            evolve_frame = cached_eoh
            history_start = len(evolve_frame._sample_order_history)
            start_sample_count = evolve_frame._tot_sample_nums
            evolve_frame.method_usage = self.algorithm_frame.method_usage(cur_method)
            evolve_frame.method_introduction = self.algorithm_frame.other_method_introductions(cur_method)
            evolve_frame.main_stream = self.algorithm_frame.main_stream()
            evolve_frame.prev_method_individual = neighbor_context["prev_method_individual"]
            evolve_frame.next_method_individual = neighbor_context["next_method_individual"]
            evolve_frame._evaluator._evaluator.algorithm_str = self.algorithm_frame.body
            evolve_frame._evaluator._evaluator.template_program = cur_template_program
            evolve_frame._template_program_str = cur_template_program
            evolve_frame._function_to_evolve = TextFunctionProgramConverter.text_to_function(cur_template_program)
            evolve_frame._template_program = TextFunctionProgramConverter.text_to_program(cur_template_program)
            evolve_frame._use_adj_prev_operator = self.use_adj_prev_operator and neighbor_context["prev_name"] is not None
            evolve_frame._use_adj_next_operator = self.use_adj_next_operator and neighbor_context["next_name"] is not None
            evolve_frame._used_operator_history = []
            evolve_frame._resume_mode = True
            evolve_frame._max_sample_nums = evolve_frame._tot_sample_nums + method_budget
            self._refresh_population_head_score(evolve_frame, self.algorithm_frame.score)
            evolve_frame.set_plot_window(
                self.total_consumed_samples,
                start_sample_count,
            )
            evolve_frame.record_round_initial_best(self.algorithm_frame.score)
            round_start_score = self.algorithm_frame.score
        success = evolve_frame.run()
        self.method_eoh_cache[cur_method] = evolve_frame
        actual_sample_count = evolve_frame._tot_sample_nums - (start_sample_count or 0)
        if history_start is not None:
            self.profiler.register_method_samples(
                cur_method,
                local_start_order=start_sample_count or 0,
                actual_sample_count=actual_sample_count,
                valid_sample_orders=evolve_frame._sample_order_history[history_start:],
                valid_sample_scores=evolve_frame._sample_score_history[history_start:],
                round_start_score=round_start_score,
            )
        best_func = evolve_frame._population.population[0]
        self.replace(cur_method, best_func)
        return best_func, evolve_frame, actual_sample_count
    
    def _run_average_schedule(self):
        baseline_score = self._evaluate_frame()
        self.algorithm_frame.score = baseline_score
        self.profiler.record_baseline(self.algorithm_frame)
        for idx, method in enumerate(self.method_order):
            budget = self._budget_for_method(method, idx)
            if budget <= 0:
                continue
            score_before = self.algorithm_frame.score
            template_program = build_template_program(self.algorithm_frame, method)
            best_func, evolve_frame, actual_sample_count = self.evolve_once(method, template_program, budget)
            self._finalize_method_round(
                method,
                budget,
                actual_sample_count,
                score_before,
                evolve_frame,
                {},
                best_func,
            )

    def _run_adaptive_schedule(self):
        baseline_score = self._evaluate_frame()
        self.algorithm_frame.score = baseline_score
        self.profiler.record_baseline(self.algorithm_frame)
        for method in self.method_order:
            if self.remaining_budget <= 0:
                return
            budget = min(self.per_call_budget_cap, self.remaining_budget)
            template_program = build_template_program(self.algorithm_frame, method)
            score_before = self.algorithm_frame.score
            best_func, evolve_frame, actual_sample_count = self.evolve_once(method, template_program, budget)
            self._finalize_method_round(
                method,
                budget,
                actual_sample_count,
                score_before,
                evolve_frame,
                {},
                best_func,
            )

        while self.remaining_budget > 0:
            method, benefit_map = self._select_next_method()
            if method is None:
                break
            budget = min(self.per_call_budget_cap, self.remaining_budget)
            template_program = build_template_program(self.algorithm_frame, method)
            score_before = self.algorithm_frame.score
            best_func, evolve_frame, actual_sample_count = self.evolve_once(method, template_program, budget)
            self._finalize_method_round(
                method,
                budget,
                actual_sample_count,
                score_before,
                evolve_frame,
                benefit_map,
                best_func,
            )
    
    def run(self):
        try:
            if self.budget_mode == "average":
                self._run_average_schedule()
            else:
                self._run_adaptive_schedule()
            test_score = self._evaluate_test_frame()
            if test_score is not None:
                self.profiler.record_test_result(
                    {
                        "test_score": test_score,
                        "num_test_instances": len(self.test_instance),
                    }
                )
            self.profiler.finish(
                self.algorithm_frame,
                {
                    "final_score": self.algorithm_frame.score,
                    "remaining_budget": self.remaining_budget,
                    "total_consumed_budget": self.total_consumed_budget,
                    "test_score": test_score,
                },
            )
            return self.algorithm_frame
        finally:
            self._close_cached_eoh()
    
class Outer:
    def __init__(self,
                 llm: HttpsApi, 
                 instance: list[tuple[np.ndarray, float]],
                 outer_pop_size: int,
                 outer_max_generations: int,
                 eoh_max_sample_nums: int,
                 eoh_pop_size: int,
                 timeout_seconds: int = 120,
                 eval_num_workers: int | None = None):
        
        self.llm = llm
        self.instance = instance
        self.max_generations = outer_max_generations
        
        self.eoh_max_sample_nums = eoh_max_sample_nums
        self.eoh_pop_size = eoh_pop_size
        self.timeout_seconds = timeout_seconds
        self.eval_num_workers = eval_num_workers
        
        self.population = FramePopulation(pop_size=outer_pop_size)
        self.profiler = FrameProfiler(log_dir="logs_frame_0620")
        self.eoh_log_root = os.path.join(self.profiler._log_dir, "inner_eoh")
        
        self.selection_num = 2
        self.tot_sample_nums = 0
        self.use_e2_operator = True
        self.use_m1_operator = True

        self.debug_mode = True

    def _evaluate_with_timeout(self, algorithm_frame: AlgorithmFrame) -> float:
        result_queue = mp.Queue()
        process = mp.Process(
            target=_frame_evaluation_worker,
            args=(
                self.instance,
                algorithm_frame.body,
                self.timeout_seconds,
                result_queue,
                "gap",
                self.eval_num_workers,
            ),
        )
        process.start()
        try:
            result = result_queue.get(timeout=self.timeout_seconds)
        except Empty:
            raise TimeoutError(f"Frame evaluation exceeded {self.timeout_seconds} seconds.")
        finally:
            _stop_process(process)
            result_queue.close()
            result_queue.join_thread()

        if "error" in result:
            raise RuntimeError(result["error"])
        return result["score"]

    def _fill_frame_metadata(
        self,
        algorithm_frame: AlgorithmFrame,
        score: float,
        sample_time: float,
        evaluate_time: float,
        operator: str,
    ) -> None:
        # Normalize frame metadata once so logging and population updates use the same values.
        algorithm_frame.score = score
        algorithm_frame.sample_time = sample_time
        algorithm_frame.evaluate_time = evaluate_time
        algorithm_frame.operator = operator
        if not algorithm_frame.frame_id:
            algorithm_frame.frame_id = f"frame_{self.tot_sample_nums:04d}_{operator}"

    def sample_evaluate_register(self, prompt, operator):
        sample_start_time = time.time()
        response_text = ""
        algorithm_frame = None
        try:
            response_text = self.llm.draw_sample(prompt) if not use_exist else response
            self.tot_sample_nums += 1
            algorithm_frame = text_to_algorithm(response_text)
            sample_time = time.time() - sample_start_time
            if algorithm_frame is None:
                return
            evaluate_start_time = time.time()
            score = self._evaluate_with_timeout(algorithm_frame)
            score = None if math.isinf(score) else score
            evaluate_time = time.time() - evaluate_start_time
            self._fill_frame_metadata(
                algorithm_frame, score, sample_time, evaluate_time, operator
            )
            if _is_valid_outer_score(score):
                self.population.register_frame(algorithm_frame)
            self.profiler.register_frame(algorithm_frame)
        except Exception as e:
            print(f"Error during algorithm frame sampling or evaluation: {e}")
            if algorithm_frame is None:
                # Preserve as much structured content as possible from the raw response before falling back.
                try:
                    algorithm_frame = text_to_algorithm(response_text) if response_text else None
                except Exception:
                    algorithm_frame = None
                if algorithm_frame is None:
                    algorithm_frame = AlgorithmFrame(
                        score=None,
                        body="",
                        class_args=[],
                        method_args={},
                        method_introduction={},
                        evaluate_time=0.0,
                        sample_time=time.time() - sample_start_time,
                        operator=operator,
                        frame_id=f"frame_{self.tot_sample_nums:04d}_{operator}",
                        response_text=response_text,
                    )
            self._fill_frame_metadata(
                algorithm_frame,
                algorithm_frame.score,
                time.time() - sample_start_time,
                algorithm_frame.evaluate_time,
                operator,
            )
            self.profiler.register_frame(algorithm_frame)
    
    def iteratively_init_population(self):
        while self.population.generation == 0 and len(self.population.init_population) < self.population.pop_size:
            if self.tot_sample_nums == 20:
                exit("Stop iterative initialization after 20 samples to prevent infinite loop. Please check the logs for details.")
            try:
                prompt = FramePrompt.get_frame_prompt_i1(task_description=task_description_outer,
                                                         problem_info=problem_info,
                                                         algorithm_template=algorithm_template)
                self.sample_evaluate_register(prompt, 'i1')
            except Exception as e:
                if self.debug_mode:
                    traceback.print_exc()
                    exit()
                continue
        print(f'Note: During initialization, FrameEoH gets {self.population.pop_size} algorithms '
              f'after {self.tot_sample_nums} trails.')
        
    def inner_evolve(self):
        for indiv in self.population.init_population:
            inner = Inner(llm=self.llm, train_instance=self.instance, test_instance=None, algorithm_frame=indiv,
                          eoh_log_root=self.eoh_log_root,
                          max_sample_nums=self.eoh_max_sample_nums, pop_size=self.eoh_pop_size,
                          eval_num_workers=self.eval_num_workers)
            inner.run()
            if _is_valid_outer_score(inner.algorithm_frame.score):
                self.population.register_frame(inner.algorithm_frame)
                self.profiler.register_frame(inner.algorithm_frame)
    
    def continue_loop(self):
        return len(self.population.init_population) < self.population.pop_size
        
    def frame_evolve(self):
        while self.continue_loop():
            try:
                # ger a new frame using e1
                indivs = [self.population.selection() for _ in range(self.selection_num)]
                prompt = FramePrompt.get_frame_prompt_e1(task_description=task_description_outer,
                                                         problem_info=problem_info,
                                                         algorithm_template=algorithm_template,
                                                         indivs=indivs)
                if self.debug_mode:
                    print(f"Frame Prompt (E1): {prompt}")
                self.sample_evaluate_register(prompt, 'e1')
                if not self.continue_loop():
                    break
                
                # ger a new frame using e2
                if self.use_e2_operator:
                    indivs = [self.population.selection() for _ in range(self.selection_num)]
                    prompt = FramePrompt.get_frame_prompt_e2(task_description=task_description_outer,
                                                             problem_info=problem_info,
                                                             algorithm_template=algorithm_template,
                                                             indivs=indivs)
                    if self.debug_mode:
                        print(f"Frame Prompt (E2): {prompt}")
                    self.sample_evaluate_register(prompt, 'e2')
                    if not self.continue_loop():
                        break
                    
                if self.use_m1_operator:
                    # ger a new frame using m1
                    indiv = self.population.selection()
                    prompt = FramePrompt.get_frame_prompt_m1(task_description=task_description_outer,
                                                             problem_info=problem_info,
                                                             algorithm_template=algorithm_template,
                                                             indiv=indiv)
                    if self.debug_mode:
                        print(f"Frame Prompt (M1): {prompt}")
                    self.sample_evaluate_register(prompt, 'm1')
                    if not self.continue_loop():
                        break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self.debug_mode:
                    traceback.print_exc()
                    exit()
                continue          
                
    def run(self):
        self.iteratively_init_population()
        self.inner_evolve()
        while self.population.generation <= self.max_generations:
            self.frame_evolve()
            self.inner_evolve()
        self.profiler.finish()


def _coords_to_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_random_train_instances(
    num_instances: int,
    city_num: int,
    seed: int = 0,
) -> list[tuple[np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(num_instances):
        coords = rng.uniform(0.0, 1.0, size=(city_num, 2))
        distance_matrix = _coords_to_distance_matrix(coords)
        instances.append((distance_matrix, 0.0))
    return instances


def build_npz_test_instances(max_city_num: int) -> list[tuple[np.ndarray, float]]:
    distance_matrix_dict, _ = load_tsp_dictionaries()
    instances = []
    for distance_matrix, optimum_distance in distance_matrix_dict.values():
        if optimum_distance is None:
            continue
        if distance_matrix.shape[0] <= max_city_num:
            instances.append((distance_matrix, float(optimum_distance)))
    return instances


def run_inner_only(
    llm: HttpsApi,
    train_instance: list[tuple[np.ndarray, float]],
    test_instance: list[tuple[np.ndarray, float]],
    train_num_instances: int,
    train_city_num: int,
    train_seed: int = 0,
    max_train_resample_trials: int = 50,
    eval_num_workers: int | None = None,
) -> AlgorithmFrame | None:
    """
    combo_settings = [
        ("absolute_gain", {"prev"}, True),
        ("absolute_gain", {"next"}, True),
        ("absolute_gain", {"prev", "next"}, True),
        ("relative_gain", {"prev"}, True),
        ("relative_gain", {"next"}, True),
        ("relative_gain", {"prev", "next"}, True),
        ("relative_discounted_gain", {"prev"}, True),
        ("relative_discounted_gain", {"next"}, True),
        ("relative_discounted_gain", {"prev", "next"}, True)
    ]
    
    combo_settings = [
        ("absolute_gain", {"prev"}, False),
        ("absolute_gain", {"next"}, False),
        ("absolute_gain", {"prev", "next"}, False),
        ("relative_gain", {"prev"}, False),
        ("relative_gain", {"next"}, False),
        ("relative_gain", {"prev", "next"}, False),
        ("relative_discounted_gain", {"prev"}, False),
        ("relative_discounted_gain", {"next"}, False),
        ("relative_discounted_gain", {"prev", "next"}, False)
    ]
    """
    
    combo_settings = [
        ("absolute_gain", {"prev"}, True)]
    last_successful_frame = None

    for benefit_mode, operator_mode, use_context_prompt in combo_settings:
        operator_tag = "_".join(sorted(operator_mode)) if operator_mode else "none"
        context_tag = "with_context" if use_context_prompt else "raw_prompt"
        print(
            f"Start combination: benefit_mode={benefit_mode}, "
            f"operator_mode={operator_tag}, context_mode={context_tag}",
            flush=True,
        )
        try:
            baseline_ready = False
            for offset in range(max_train_resample_trials):
                # Resample the training set until this combination gets a valid baseline.
                current_train_instance = train_instance
                if offset > 0:
                    current_train_instance = build_random_train_instances(
                        train_num_instances,
                        train_city_num,
                        seed=train_seed + offset,
                    )
                algorithm_frame = text_to_algorithm(response)
                algorithm_frame.frame_id = "inner_only_frame"
                inner = Inner(
                    llm=llm,
                    train_instance=current_train_instance,
                    test_instance=test_instance,
                    algorithm_frame=algorithm_frame,
                    eoh_log_root=f"experiment_average_{context_tag}",
                    max_sample_nums=1000,
                    pop_size=10,
                    budget_mode="average",
                    benefit_mode=benefit_mode,
                    method_selection_mode="softmax",
                    per_call_budget_cap=50,
                    discount_factor=0.8,
                    softmax_temperature=1.0,
                    use_adj_prev_operator=False,
                    use_adj_next_operator=False,
                    use_context_prompt=use_context_prompt,
                    eval_num_workers=eval_num_workers,
                )
                baseline_score = inner._evaluate_frame()
                if baseline_score is None:
                    print(
                        f"Skip current train seed {train_seed + offset} for "
                        f"{benefit_mode}/{operator_tag}/{context_tag}: baseline score is None.",
                        flush=True,
                    )
                    continue

                inner.algorithm_frame.score = baseline_score
                print(
                    f"Initial frame baseline score: {baseline_score} "
                    f"(train_seed={train_seed + offset})",
                    flush=True,
                )
                last_successful_frame = inner.run()
                baseline_ready = True
                break

            if not baseline_ready:
                print(
                    f"Skip combination benefit_mode={benefit_mode}, "
                    f"operator_mode={operator_tag}, context_mode={context_tag}: failed to obtain a valid "
                    f"baseline after {max_train_resample_trials} trials.",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"Skip combination benefit_mode={benefit_mode}, "
                f"operator_mode={operator_tag}, context_mode={context_tag}: {exc}",
                flush=True,
            )
            traceback.print_exc()

    return last_successful_frame
    
def main() -> None:
    mode = "inner_only"
    train_city_num = 100
    train_num_instances = 50
    train_seed = 0
    test_max_city_num = 100
    eval_num_workers = 20
    train_instance = build_random_train_instances(train_num_instances, train_city_num, seed=train_seed)
    test_instance = build_npz_test_instances(test_max_city_num)

    llm = HttpsApi(
        host="yunwu.ai",
        key="sk-B59wA4vTQe4AXRhHM9tg4sleZ3amBS8jtuVSJ1ambUuVmnls",
        model="gpt-4o-mini",
        timeout=120,
    )

    if mode == "inner_only":
        run_inner_only(
            llm,
            train_instance,
            test_instance,
            train_num_instances=train_num_instances,
            train_city_num=train_city_num,
            train_seed=train_seed,
            eval_num_workers=eval_num_workers,
        )
        return

    outer = Outer(llm=llm, 
                  instance=test_instance, 
                  outer_pop_size=5, 
                  outer_max_generations=10,
                  eoh_max_sample_nums=100,
                  eoh_pop_size=10,
                  eval_num_workers=eval_num_workers)
    outer.run()
if __name__ == "__main__":
    main()
