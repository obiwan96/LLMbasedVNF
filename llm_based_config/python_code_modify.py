import ast
import astor  # Import astor to convert AST back to source code
import os
#################################################
# modify python code.                           #
# put all 'top-level' into '__main__'.          #
# So that prevent the test program fall into    #
# an infinite loop accidentally created by LLM. #
#        Automatically made by GPT-4.o          #
#################################################

def is_top_level_node(node):
    """
    Check if a node is a top-level node.
    We treat functions, classes, and imports as non-top-level code.
    """
    return isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))

def wrap_code_in_main(file_path, output_file_path=None):
    """
    Read the given Python file, wrap top-level code in the `if __name__ == "__main__":` block,
    and write the modified code to a new file (or print it).
    """
    # Read the Python code from the file
    with open(file_path, 'r') as f:
        code = f.read()

    # Parse the code into an AST
    try:
        tree = ast.parse(code)
    except:
        return False

    # Separate top-level code (assignments, expressions) from functions and classes
    top_level_code = []
    non_top_level_code = []

    for node in tree.body:
        if is_top_level_node(node):
            non_top_level_code.append(node)
        else:
            top_level_code.append(node)

    # Convert AST nodes back into code using astor
    non_top_level_code_str = astor.to_source(ast.Module(body=non_top_level_code, type_ignores=[]))

    if top_level_code:
        top_level_code_str = astor.to_source(ast.Module(body=top_level_code, type_ignores=[]))

        # Create the modified code with `if __name__ == "__main__":` wrapping
        modified_code = non_top_level_code_str + '\n\nif __name__ == "__main__":\n'
        
        # Indent the top-level code
        indented_top_level_code = '\n'.join(['    ' + line for line in top_level_code_str.splitlines()])
        modified_code += indented_top_level_code
    else:
        # No top-level code, just return the non-top-level code
        modified_code = non_top_level_code_str.strip()  # Remove any trailing newlines
    # Write to the output file if provided, else print
    if output_file_path:
        with open(output_file_path, 'w') as f:
            f.write(modified_code)
        #print(f"Modified code written to {output_file_path}")
    else:
        print(modified_code)
    return True

if __name__ == "__main__":
    # Example usage
    file_to_modify = "Bad_Example/infinite_loop.py"  # Replace with your actual file
    output_file = "Bad_Example/infinite_loop_new.py"  # The modified file (optional)

    wrap_code_in_main(file_to_modify, output_file_path=output_file)