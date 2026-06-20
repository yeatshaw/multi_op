from pathlib import Path
import ast
from dataclasses import dataclass
import re
from threading import Lock
import textwrap
import numpy as np

@dataclass
class AlgorithmFrame:
    score: float
    body: str
    class_args: list[str]
    method_args: dict[str, str]
    method_introduction: dict[str, str]
    evaluate_time: float
    sample_time: float
    operator: str = "Unknown"
    frame_id: str = ""
    response_text: str = ""

    def method_usage(self, method_name):
        return self.method_introduction.get(method_name, None)
    
    def other_method_introductions(self, method_name):
        return "\n".join(
            f"- {name}: {intro}"
            for name, intro in self.method_introduction.items()
            if name != method_name
        )
        
    def main_stream(self):
        tree = ast.parse(self.body)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Algorithm":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "run":
                        src = ast.get_source_segment(self.body, item)
                        lines = src.splitlines()
                        if len(lines) <= 1:
                            return src.strip() + "\n"
                        body = textwrap.indent(textwrap.dedent("\n".join(lines[1:])), "    ")
                        return "\n".join([lines[0], body]).strip() + "\n"
        return ""
    
    def description(self):
        method_lines = "\n".join(
            f"- {name}: {intro}" for name, intro in self.method_introduction.items()
        )
        class_args_lines = "\n".join(self.class_args)
        return (
            "main_workflow:\n"
            f"{self.main_stream().rstrip()}\n\n"
            "method_introduction:\n"
            f"{method_lines}\n\n"
            "Class_args:\n"
            f"{class_args_lines}\n"
        )
        
class FramePopulation:
    def __init__(self, 
                 pop_size, 
                 generation=0,
                 pop: list[AlgorithmFrame] | None = None):
        if  isinstance(pop, list):
            self.population = pop
        elif pop is None:
            self.population = []
        
        self.init_population = []
        self.pop_size = pop_size
        self.lock = Lock()
        self.next_population = []
        self.generation = generation
    
    def __len__(self):
        return len(self.population)
    
    def __getitem__(self, item):
        return self.population[item]
    
    def __setitem__(self, key, value):
        self.population[key] = value
        
    def survival(self):
        pop = self.population + self.next_population
        pop = sorted(pop, key=lambda al: al.score, reverse=True)
        self.population = pop[:self.pop_size]
        self.init_population = []
        self.next_population = []
        self.generation += 1
    
    def register_frame(self, frame: AlgorithmFrame):
        if len(self.init_population) < self.pop_size and frame.score is None:
            return
        if len(self.init_population) < self.pop_size and frame.score is not None:
            self.lock.acquire()
            # Only keep unique frames in the initial population to ensure diversity
            if not any(frame.body == f.body for f in self.init_population):
                self.init_population.append(frame)
            self.lock.release()
            return
        try:
            self.lock.acquire()
            self.next_population.append(frame)
            if len(self.next_population) >= self.pop_size:
                self.survival()
        except Exception as e:
            return
        finally:
            self.lock.release()
        
    def selection(self) -> AlgorithmFrame:
        als = [al for al in self.population if al.score is not None]
        al = sorted(als, key=lambda f: f.score, reverse=True)
        p = [1 / (r + len(al)) for r in range(len(al))]
        p = np.array(p) / sum(p)
        return np.random.choice(al, p=p)

def load_tsp_dictionaries(
    input_file: Path = Path("/public/home/liuyang/multi_op/tsp_instances.npz"),
) -> tuple[dict[str, tuple[np.ndarray, int]], dict[str, tuple[np.ndarray, int]]]:
    with np.load(input_file, allow_pickle=True) as data:
        distance_matrix_dict = data["distance_matrix_dict"].item()
        coordinate_dict = data["coordinate_dict"].item()
    return distance_matrix_dict, coordinate_dict


def text_to_algorithm(response) -> AlgorithmFrame:
    response_text = getattr(response, "text", response)
    body = _extract_code(response_text)
    class_args_text = _extract_braced_section(response_text, "Class_args")
    method_args_text = _extract_braced_section(response_text, "method_args")
    method_text = _extract_braced_section(response_text, "method")
    valid_methods = _extract_evolvable_methods(body)
    return AlgorithmFrame(
        score=float("-inf"),
        body=body,
        class_args=_split_non_empty_lines(class_args_text),
        method_args=_extract_method_args(method_args_text, valid_methods),
        method_introduction=_extract_method_introduction(method_text),
        evaluate_time=0.0,
        sample_time=0.0,
        operator="Unknown",
        frame_id="",
        response_text=response_text,
    )


def build_template_program(parsed: AlgorithmFrame, method_name: str) -> str:
    code = parsed.body
    tree = ast.parse(code)
    imports, method_src = [], ""

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(code, node))
        elif isinstance(node, ast.ClassDef) and node.name == "Algorithm":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    method_src = ast.get_source_segment(code, item)
                    break

    if not method_src:
        return ""

    signature = method_src.split(":\n", 1)[0] + ":"
    method_info = parsed.method_args.get(method_name, "")
    arg_block, return_block = _split_method_info(method_info)
    doc = [
        '    """',
        "    Arg:",
        "        self: The instance of the class.",
        *[f"            {line}" for line in _expand_class_args(parsed.class_args)],
        *[f"        {line}" for line in arg_block],
    ]
    if return_block:
        doc += ["    Returns:", *[f"        {line}" for line in return_block]]
    doc.append('    """')
    program = (
        "\n".join(imports)
        + "\n\n"
        + signature
        + "\n"
        + "\n".join(doc)
        + "\n"
    )
    return textwrap.dedent(program).strip() + "\n"


def _extract_code(response_text: str) -> str:
    # The current prompt always puts `Code:` inside a python fence and wraps the body in braces.
    match = re.search(r"```(?:python)?\s*Code:\s*(.*?)```", response_text, flags=re.DOTALL)
    if match:
        content = _strip_wrapping_braces(match.group(1))
        if "class Algorithm:" in content:
            return "\n".join(line.rstrip() for line in content.splitlines() if line.strip())
    return ""


def _extract_braced_section(response_text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*\{{(.*?)\}}", response_text, flags=re.DOTALL)
    if not match:
        return ""
    return textwrap.dedent(match.group(1)).strip()


def _extract_method_args(method_args_text: str, valid_methods: set[str] | None = None) -> dict[str, str]:
    if not method_args_text:
        return {}
    parts = re.split(r"\n\s{4}([A-Za-z_]\w*):\s*\n", "\n" + method_args_text.strip("\n"))
    method_args = {
        parts[i]: textwrap.dedent(parts[i + 1]).strip()
        for i in range(1, len(parts), 2)
    }
    if valid_methods is None:
        return {
            name: info for name, info in method_args.items()
            if name not in {"run", "__init__"}
        }
    return {
        name: info for name, info in method_args.items()
        if name in valid_methods
    }


def _extract_evolvable_methods(code: str) -> set[str]:
    methods = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Algorithm":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name not in {"run", "__init__"}:
                    methods.add(item.name)
    return methods


def _expand_class_args(class_args: list[str]) -> list[str]:
    expanded = []
    for line in class_args:
        expanded.append(line)
        if "tour_evaluation_function" in line:
            expanded.extend(_tour_eval_signature_lines())
    return expanded


def _tour_eval_signature_lines() -> list[str]:
    return [
        "    args:",
        "        tour: list[int], the visiting order of cities, where the first city is the starting city.",
        "        distance_matrix: np.ndarray in the shape of (city_num, city_num), the pairwise city distance matrix.",
        "    Return:",
        "        total_distance: float, the total tour length.",
    ]


def _extract_method_introduction(content: str) -> dict[str, str]:
    result = {}
    for line in content.splitlines():
        line = line.strip()
        match = re.match(r"-\s*'?([A-Za-z_]\w*)'?\s*:\s*(.*)", line)
        if match:
            result[match.group(1)] = match.group(2).strip()
    return result


def _split_non_empty_lines(content: str) -> list[str]:
    return [line.rstrip() for line in content.splitlines() if line.strip()]


def _strip_wrapping_braces(content: str) -> str:
    lines = [line.rstrip() for line in content.strip().splitlines()]
    if lines and lines[0].strip() == "{":
        lines = lines[1:]
    if lines and lines[-1].strip() == "}":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _split_method_info(method_info: str) -> tuple[list[str], list[str]]:
    arg_match = re.search(r"Arg:\s*(.*?)(?=\nReturn:|\nReturns:|\Z)", method_info, flags=re.DOTALL)
    ret_match = re.search(r"Returns?:\s*(.*)", method_info, flags=re.DOTALL)
    arg_block = [_strip_leading_dash(line.rstrip()) for line in (arg_match.group(1).splitlines() if arg_match else []) if line.strip() and line.strip() != "- None"]
    return_block = [_strip_leading_dash(line.rstrip()) for line in (ret_match.group(1).splitlines() if ret_match else []) if line.strip() and line.strip() != "- None"]
    return arg_block, return_block


def _strip_leading_dash(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("- "):
        indent = len(line) - len(line.lstrip())
        return " " * indent + stripped[2:]
    return line
