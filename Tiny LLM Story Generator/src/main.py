from model_loader import load_model
from filter import is_python_question
from generator import generate_response

NON_PYTHON_MESSAGE = (
    "❌ I can only answer questions related to Python programming."
)

def main():
    model, tokenizer, device = load_model()

    print("Python-Only GPT-2 (type 'exit' to quit)\n")

    while True:
        user_input = input(">> ")

        if user_input.lower() == "exit":
            break

        if is_python_question(user_input):
            response = generate_response(
                model, tokenizer, device, user_input
            )
            print("\n", response, "\n")
        else:
            print("\n", NON_PYTHON_MESSAGE, "\n")

if __name__ == "__main__":
    main()
