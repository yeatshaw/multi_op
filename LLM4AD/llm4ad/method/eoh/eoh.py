# Module Name: EoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang.
#       "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model."
#       In Forty-first International Conference on Machine Learning (ICML). 2024.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import math
import os
import time
import traceback
from threading import Thread
from typing import Optional, Literal

from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase


class EoH:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 300,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 use_adj_prev_operator: bool = False,
                 use_adj_next_operator: bool = False,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator
        self._use_adj_prev_operator = use_adj_prev_operator
        self._use_adj_next_operator = use_adj_next_operator

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = EoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            2 * self._pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

        self.method_name = evaluation.method_name
        self.method_usage = kwargs.get("method_usage", None)
        self.method_introduction = kwargs.get("method_introduction", None)
        self.main_stream = kwargs.get("main_stream", None)
        self.prev_method_name = kwargs.get("prev_method_name", None)
        self.prev_method_code = kwargs.get("prev_method_code", None)
        self.prev_method_thought = kwargs.get("prev_method_thought", None)
        self.next_method_name = kwargs.get("next_method_name", None)
        self.next_method_code = kwargs.get("next_method_code", None)
        self.next_method_thought = kwargs.get("next_method_thought", None)
        self._sample_score_history = []
        self._sample_order_history = []
        self._used_operator_history = []
        
    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _sample_evaluate_register(self, prompt, operator):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()
        # register to profiler
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator
        self._used_operator_history.append(operator)
        if score is not None and not math.isinf(score) and score <= 0:
            self._sample_order_history.append(self._tot_sample_nums + 1)
            self._sample_score_history.append(score)
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1

        # register to the population
        self._population.register_function(func)

    def _plot_convergence_curve(self):
        if not self._sample_score_history:
            return
        log_dir = getattr(self._profiler, "_log_dir", None)
        if not log_dir:
            return
        try:
            import matplotlib.pyplot as plt

            best_so_far = []
            current_best = float('-inf')
            for score in self._sample_score_history:
                current_best = max(current_best, score)
                best_so_far.append(current_best)

            plt.figure(figsize=(8, 5))
            plt.plot(self._sample_order_history, best_so_far, marker='o')
            plt.xlabel('Sample Order')
            plt.ylabel('Best Score So Far')
            plt.title(f'EoH Best Score Curve - {self.method_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'convergence_{self.method_name or "method"}.png'
            plt.savefig(os.path.join(log_dir, filename), dpi=150)
            plt.close()
        except Exception:
            if self._debug_mode:
                traceback.print_exc()

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_eoh_operator(self):
        while self._continue_loop():
            try:
                # get a new func using e1
                indivs = [self._population.selection() for _ in range(self._selection_num)]
                prompt = EoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve, 
                                                 self.method_name, self.method_usage, self.method_introduction,
                                                 self.main_stream)
                if self._debug_mode:
                    print(f'E1 Prompt: {prompt}')
                self._sample_evaluate_register(prompt, 'e1')
                if not self._continue_loop():
                    break

                # get a new func using e2
                if self._use_e2_operator:
                    indivs = [self._population.selection() for _ in range(self._selection_num)]
                    prompt = EoHPrompt.get_prompt_e2(self._task_description_str, indivs, self._function_to_evolve, 
                                                     self.method_name, self.method_usage, self.method_introduction,
                                                     self.main_stream)
                    if self._debug_mode:
                        print(f'E2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, 'e2')
                    if not self._continue_loop():
                        break

                # get a new func using m1
                if self._use_m1_operator:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve, 
                                                     self.method_name, self.method_usage, self.method_introduction,
                                                     self.main_stream)
                    if self._debug_mode:
                        print(f'M1 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, 'm1')
                    if not self._continue_loop():
                        break

                # get a new func using m2
                if self._use_m2_operator:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve, 
                                                     self.method_name, self.method_usage, self.method_introduction,
                                                     self.main_stream)
                    if self._debug_mode:
                        print(f'M2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, 'm2')
                    if not self._continue_loop():
                        break

                if self._use_adj_prev_operator and self.prev_method_name and self.prev_method_code:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_adj_prev(
                        self._task_description_str,
                        indiv,
                        self._function_to_evolve,
                        self.method_name,
                        self.method_usage,
                        self.method_introduction,
                        self.main_stream,
                        self.prev_method_name,
                        self.prev_method_code,
                        self.prev_method_thought,
                    )
                    if self._debug_mode:
                        print(f'AP Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, 'ap')
                    if not self._continue_loop():
                        break

                if self._use_adj_next_operator and self.next_method_name and self.next_method_code:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_adj_next(
                        self._task_description_str,
                        indiv,
                        self._function_to_evolve,
                        self.method_name,
                        self.method_usage,
                        self.method_introduction,
                        self.main_stream,
                        self.next_method_name,
                        self.next_method_code,
                        self.next_method_thought,
                    )
                    if self._debug_mode:
                        print(f'AN Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, 'an')
                    if not self._continue_loop():
                        break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    #exit()
                continue

    def _iteratively_init_population(self):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            try:
                # get a new func using i1
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve, 
                                                 self.method_name, self.method_usage, self.method_introduction,
                                                 self.main_stream)
                self._sample_evaluate_register(prompt, 'i1')
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    # print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    print(
                        f'Note: During initialization, EoH gets {len(self._population) + len(self._population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    #exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def _shutdown_evaluation_executor(self):
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except Exception:
            pass

    def run(self) -> bool:
        try:
            if not self._resume_mode:
                # do initialization
                self._multi_threaded_sampling(self._iteratively_init_population)
                self._population.survival()
                # terminate searching if
                if len(self._population) < self._selection_num:
                    print(
                        f'The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. '
                        f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                        f'Please also check your evaluation implementation and LLM implementation.')
                    return False

            # evolutionary search
            self._multi_threaded_sampling(self._iteratively_use_eoh_operator)

            # finish
            self._plot_convergence_curve()
            if self._profiler is not None:
                self._profiler.finish()
            return len(self._population) > 0
        finally:
            self._shutdown_evaluation_executor()
            self._sampler.llm.close()
