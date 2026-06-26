from __future__ import annotations
    
import os
import sys
import ast
import math
import time
import json
import textwrap
import traceback
import logging
import multiprocessing as mp
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
from LLM4AD.llm4ad.tools.llm.llm_api_https import HttpsApi
from LLM4AD.llm4ad.tools.profiler import ProfilerBase
from LLM4AD.llm4ad.method.eoh import EoH

use_exist = False


def _is_valid_outer_score(score: float | None) -> bool:
    return score is not None and not math.isinf(score) and score <= 0

def _frame_evaluation_worker(
    instance: list[tuple[np.ndarray, float]],
    algorithm_body: str,
    timeout_seconds: int,
    result_queue: mp.Queue,
) -> None:
    try:
        evaluation = TSPEvaluation(
            instance=instance,
            algorithm_str=algorithm_body,
            timeout_seconds=timeout_seconds,
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
                 instance: list[tuple[np.ndarray, float]],
                 algorithm_frame: AlgorithmFrame,
                 eoh_log_root: str,
                 max_sample_nums: int = 20,
                 pop_size: int = 5):
        self.llm = llm
        self.instance = instance
        self.algorithm_frame = algorithm_frame
        self.eoh_log_root = eoh_log_root
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
    
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
        
    def initial_evolve(self):
        for method in self.algorithm_frame.method_args.keys():
            template_program = build_template_program(self.algorithm_frame, method)
            self.evolve_once(method, template_program)
    
    def evolve_once(self, cur_method, cur_template_program):
        frame_log_dir = os.path.join(
            self.eoh_log_root,
            self.algorithm_frame.frame_id or "unknown_frame",
            cur_method,
        )
        evaluation = TSPEvaluation(template_program=cur_template_program,
                                   task_description=task_description_inner,
                                   instance=self.instance, 
                                   method_name=cur_method,
                                   algorithm_str=self.algorithm_frame.body)
        evolve_frame = EoH(llm=self.llm,
                           profiler=ProfilerBase(log_dir=frame_log_dir, log_style='complex'),
                           evaluation=evaluation,
                           max_sample_nums=self.max_sample_nums,
                           max_generations=100,
                           pop_size=self.pop_size,
                           num_samplers=self.pop_size,
                           num_evaluators=self.pop_size,
                           method_usage=self.algorithm_frame.method_usage(cur_method),
                           method_introduction=self.algorithm_frame.other_method_introductions(cur_method),
                           debug_mode=True)
        success = evolve_frame.run()
        if not success or len(evolve_frame._population) == 0:
            print(
                f"Skip method '{cur_method}' for {self.algorithm_frame.frame_id or 'unknown_frame'} "
                f"because EoH failed to initialize enough feasible individuals."
            )
            return
        self.algorithm_frame.score = evolve_frame._population._population[0].score
        self.replace(cur_method, evolve_frame._population[0])
        
    def run(self):
        self.initial_evolve()
        pass
    
class Outer:
    def __init__(self,
                 llm: HttpsApi, 
                 instance: list[tuple[np.ndarray, float]],
                 outer_pop_size: int,
                 outer_max_generations: int,
                 eoh_max_sample_nums: int,
                 eoh_pop_size: int,
                 timeout_seconds: int = 120):
        
        self.llm = llm
        self.instance = instance
        self.max_generations = outer_max_generations
        
        self.eoh_max_sample_nums = eoh_max_sample_nums
        self.eoh_pop_size = eoh_pop_size
        self.timeout_seconds = timeout_seconds
        
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
            args=(self.instance, algorithm_frame.body, self.timeout_seconds, result_queue),
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
            response_text = self.llm.draw_sample(prompt) if not use_exist else response1
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
            inner = Inner(llm=self.llm, instance=self.instance, algorithm_frame=indiv,
                          eoh_log_root=self.eoh_log_root,
                          max_sample_nums=self.eoh_max_sample_nums, pop_size=self.eoh_pop_size)
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
    
def main() -> None:
    
    problem_size = 100
    instance = []
    distance_matrix_dict, _ = load_tsp_dictionaries()
    #instance.append(distance_matrix_dict['bays29'])
    for value in distance_matrix_dict.values():
        if value[0].shape[0] <= problem_size:
            instance.append(value)

    llm = HttpsApi(
        host="yunwu.ai",
        key="sk-B59wA4vTQe4AXRhHM9tg4sleZ3amBS8jtuVSJ1ambUuVmnls",
        model="gpt-4o-mini",
        timeout=120,
    )
    
    outer = Outer(llm=llm, 
                  instance=instance, 
                  outer_pop_size=5, 
                  outer_max_generations=10,
                  eoh_max_sample_nums=100,
                  eoh_pop_size=10)
    outer.run()
if __name__ == "__main__":
    main()
