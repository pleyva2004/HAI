from backend.services import claude
from toon_format import encode, decode
from backend.workflows.state import UserInput, HAIState
import base64
import json


def convert_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():

    image_path= "backend/examples/Example Query.png"
    image_base64 = convert_image_to_base64(image_path)

    # Mock API request
    mock_API_request = json.dumps({
        "user_image": image_base64,
        "requested_section": "Math",
        "requested_difficulty": "Hard",
        "requested_domain": "Problem-Solving and Data Analysis"
    })

    # Convert JSON string to dict
    mock_API_dict = json.loads(mock_API_request)
    # Create a UserInput object from the dict
    user_input = UserInput(**mock_API_dict)

    # Create initial state
    graph_state = HAIState.create_initial_state(user_input)

    print("\n\n")
    try:
        extracted_data = claude.extract_from_image(image_base64)

        print("Extracted data:")
        print("--------------------------------")
        print(extracted_data)
        print("--------------------------------")

        print("\n\n")

        encoded_extracted_data = encode(extracted_data.model_dump())
        print("Extracted data Dump TOON:")
        print("--------------------------------")
        print(encoded_extracted_data)
        print("--------------------------------")

        classified_data = claude.classify_question(encoded_extracted_data)
        print("Classified data:")
        print("--------------------------------")
        print(classified_data)
        print("--------------------------------")

        print("\n\n")

        encoded_classified_data = encode(classified_data.model_dump())
        print("Classified data Dump:")
        print("--------------------------------")
        print(encoded_classified_data)
        print("--------------------------------")

        print("\n\n")
        graph_state.extracted_features = extracted_data
        generated_question = claude.generate_question(encoded_extracted_data, encoded_classified_data)

        print("Generated question:")
        print("--------------------------------")
        print(generated_question)
        print("--------------------------------")

        print("\n\n")

        encoded_generated_question = encode(generated_question.model_dump())
        print("Generated question Dump TOON:")
        print("--------------------------------")
        print(encoded_generated_question)
        print("--------------------------------")

        print("\n\n")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
