
class FramePrompt:
    
    @classmethod
    def get_frame_prompt_i1(cls, 
                            task_description: str,
                            problem_info: str,
                            algorithm_template: str,
                            ) -> str:
        return f'''{task_description}
This is the template for the content you need to output:
{algorithm_template}

Requirements:
1.The algorithm should be designed in the form of a python Class called 'Algorithm', the method 'run' is its calling interface.
2.Complete the whole algorithm process by calling other methods inside 'run', other methods cannot call each other
3.Add the complete runnable code after 'Code:'.
4.Explain the function of each method after 'method:' according to the algorithm process.
5.Explain the purpose of each variable in the class after 'Class_args:'.
6.Explain the role of the input and output parameters in each method except for 'run' after 'method_args:'.
7.Keep the following identifiers from the template in the reply: 'Code:', 'method:', 'Class_args:', 'method_args:'.
8.'method_name' and 'arg' are just placeholders, you must rename them meaningfully.
9.The number of non-run methods is not fixed.
The input parameter information of the Class is: {problem_info}
Do not give additional explanations.
'''

    @classmethod
    def get_frame_prompt_e1(cls,
                            task_description: str,
                            problem_info: str,
                            algorithm_template: str,
                            indivs: list[dict[str, object]],
                            ) -> str:
        indivs_prompt = ''
        for i, indiv in enumerate(indivs):
            indivs_prompt += f'Algorithm {i + 1}\'s main workflow and related information are as follows:\n{indiv.description()}\n'
        return f'''{task_description}
I have {len(indivs)} existing algorithms with their codes as follows: {indivs_prompt}
Please help me create a new algorithm that has a totally different main workflow from the given ones. 

This is the template for the content you need to output, '[]' is a placeholder, indicating content that needs to be filled in:
{algorithm_template}

Requirements:
1.The algorithm should be designed in the form of a python Class called 'Algorithm', the method 'run' is its calling interface.
2.Complete the whole algorithm process by calling other methods inside 'run', other methods cannot call each other
3.Add the complete runnable code after 'Code:'.
4.Explain the function of each method after 'method:' according to the algorithm process.
5.Explain the purpose of each variable in the class after 'Class_args:'.
6.Explain the role of the input and output parameters in each method except for 'run' after 'method_args:'.
7.Keep these labels still: 'Code:', 'method:', 'Class_args:', 'method_args:'.

The input parameter information of the Class is:{problem_info}
Do not give additional explanations.
'''

    @classmethod
    def get_frame_prompt_e2(cls,
                            task_description: str,
                            problem_info: str,
                            algorithm_template: str,
                            indivs: list[dict[str, object]],
                            ) -> str:
        indivs_prompt = ''
        for i, indiv in enumerate(indivs):
            indivs_prompt += f'Algorithm {i + 1}\'s main workflow and related information are as follows:\n{indiv.description()}\n'
        return f'''{task_description}
I have {len(indivs)} existing algorithms with their codes as follows:
[{indivs_prompt}]
Please help me create a new algorithm that has a totally different main workflow from the given ones but can be motivated from them.

This is the template for the content you need to output, '[]' is a placeholder, indicating content that needs to be filled in:
{algorithm_template}

Requirements:
1.The algorithm should be designed in the form of a python Class called 'Algorithm', the method 'run' is its calling interface.
2.Complete the whole algorithm process by calling other methods inside 'run', other methods cannot call each other
3.Add the complete runnable code after 'Code:'.
4.Explain the function of each method after 'method:' according to the algorithm process.
5.Explain the purpose of each variable in the class after 'Class_args:'.
6.Explain the role of the input and output parameters in each method except for 'run' after 'method_args:'.

The input parameter information of the Class is:{problem_info}
Do not give additional explanations.
'''

    @classmethod
    def get_frame_prompt_m1(cls,
                            task_description: str,
                            problem_info: str,
                            algorithm_template: str,
                            indiv: dict[str, object],
                            ) -> str:
        return f'''{task_description}
I have one algorithm with its main workflow and related information are as follows:
[{indiv.description()}]
Please assist me in creating a new algorithm that has a different main workflow but can be a modified version of the algorithm provided.

This is the template for the content you need to output, '[]' is a placeholder, indicating content that needs to be filled in:
{algorithm_template}

Requirements:
1.The algorithm should be designed in the form of a python Class called 'Algorithm', the method 'run' is its calling interface.
2.Complete the whole algorithm process by calling other methods inside 'run', other methods cannot call each other
3.Add the complete runnable code after 'Code:'.
4.Explain the function of each method after 'method:' according to the algorithm process.
5.Explain the purpose of each variable in the class after 'Class_args:'.
6.Explain the role of the input and output parameters in each method except for 'run' after 'method_args:'.

The input parameter information of the Class is:{problem_info}
Do not give additional explanations.
''' 