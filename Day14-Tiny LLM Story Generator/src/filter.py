import re

PYTHON_KEYWORDS = [
    "python", "function", "list", "dict", "tuple", "loop",
    "for", "while", "if", "else", "class", "object",
    "lambda", "decorator", "exception", "try", "except",
    "pandas", "numpy", "flask", "django", "fastapi"
]

def is_python_question(prompt: str) -> bool:
    prompt = prompt.lower()

    for keyword in PYTHON_KEYWORDS:
        if re.search(rf"\b{keyword}\b", prompt):
            return True

    return False
