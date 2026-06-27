from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
    @classmethod
    def _build_context_block(
        cls,
        method_name: str,
        method_usage: str,
        method_introduction: str,
        main_stream: str | None,
    ) -> str:
        main_flow = main_stream.strip() if main_stream else "N/A"
        other_methods = method_introduction.strip() if method_introduction else "N/A"
        usage = method_usage or "Make this method compatible with the Algorithm class."
        return (
            f"You need to implement the method '{method_name}' within it, {usage}\n"
            f"The main workflow of the whole Algorithm class is:\n{main_flow}\n"
            f"The process of the other methods is introduced as follows:\n{other_methods}\n"
            "The evolved method must stay compatible with the main workflow and the other methods.\n"
        )

    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function, method_name: str, method_usage: str,
                      method_introduction: str, main_stream: str | None = None):
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function, method_name: str,
                      method_usage: str, method_introduction: str, main_stream: str | None = None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function, method_name: str,
                      method_usage: str, method_introduction: str, main_stream: str | None = None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function, method_name: str,
                      method_usage: str, method_introduction: str, main_stream: str | None = None):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # create prmpt content
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function, method_name: str,
                      method_usage: str, method_introduction: str, main_stream: str | None = None):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prmpt content
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_adj_prev(
        cls,
        task_prompt: str,
        indi: Function,
        indi_prev: Function,
        template_function: Function,
        method_name: str,
        method_usage: str,
        method_introduction: str,
        main_stream: str | None = None,
    ):
        assert hasattr(indi, 'algorithm')
        assert hasattr(indi_prev, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
The previous method that feeds the current method is:
{indi_prev.algorithm}
Code:
{str(indi_prev)}

The current parent method is:
Thought: {indi.algorithm}
Code:
{str(indi)}

Please create a new implementation for the current method that stays compatible with the previous method while improving the overall algorithm. You should explicitly consider the conceptual continuity between the two methods, so that the role of the current method naturally follows the role of the previous one instead of switching to an incoherent objective.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_adj_next(
        cls,
        task_prompt: str,
        indi: Function,
        indi_next: Function,
        template_function: Function,
        method_name: str,
        method_usage: str,
        method_introduction: str,
        main_stream: str | None = None,
    ):
        assert hasattr(indi, 'algorithm')
        assert hasattr(indi_next, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        context_block = cls._build_context_block(method_name, method_usage, method_introduction, main_stream)
        prompt_content = f'''{task_prompt}
{context_block}
The current parent method is:
Thought: {indi.algorithm}
Code:
{str(indi)}

The next method that consumes the output of the current method is:
{indi_next.algorithm}
Code:
{str(indi_next)}

Please create a new implementation for the current method that stays compatible with the next method while improving the overall algorithm.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

"""You need to implement the method '{method_name}' within it.
The process of the other methods is introduced as follows: {method_introduction}."""
