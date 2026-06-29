from __future__ import annotations

import re
from typing import Any
from typing import Tuple

from ...base import LLM, SampleTrimmer, Function, Program, TextFunctionProgramConverter


class EoHSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = self.__class__.trim_code_from_response(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function

    def get_response_thought_and_function(self, prompt: str) -> Tuple[str, str, Function]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = self.__class__.trim_code_from_response(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return response, thought, function

    def analyze_response(self, prompt: str) -> dict[str, Any]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = self.__class__.trim_code_from_response(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        program = SampleTrimmer.sample_to_program(code, self._template_program)

        issues = []
        if not response or not response.strip():
            issues.append("response_empty")
        if thought is None:
            issues.append("thought_field_missing")
        if code is None:
            issues.append("code_field_extract_failed")
        elif not code.strip():
            issues.append("code_field_empty")
        if function is None:
            issues.append("function_object_build_failed")
        if program is None:
            issues.append("program_build_failed")

        return {
            "response": response,
            "thought": thought,
            "code": code,
            "function": function,
            "program": program,
            "issues": issues,
        }

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            match = re.search(
                r'thought\s*:\s*(.+?)(?:\n\s*\n|\n```|\Z)',
                response,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match is None:
                return None
            thought = match.group(1).strip()
            if thought.startswith("{") and thought.endswith("}"):
                thought = thought[1:-1].strip()
            return thought or None
        except Exception:
            return None

    @classmethod
    def trim_code_from_response(cls, response: str) -> str | None:
        try:
            fenced_match = re.search(
                r'```python\s*(.*?)```',
                response,
                flags=re.IGNORECASE | re.DOTALL,
            )
            code_block = fenced_match.group(1) if fenced_match else response
            code_match = re.search(
                r'code\s*:\s*(.*)',
                code_block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            code_text = code_match.group(1) if code_match else code_block
            code_text = code_text.strip()
            if code_text.startswith("{") and code_text.endswith("}"):
                code_text = code_text[1:-1].strip()
            return SampleTrimmer.trim_preface_of_function(code_text)
        except Exception:
            return None
