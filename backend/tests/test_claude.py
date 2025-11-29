from backend.services import claude
import base64


def convert_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

if __name__ == "__main__":

    image_path= "backend/examples/Example Query.png"
    image_base64 = convert_image_to_base64(image_path)

    try:
        extracted_data = claude.extract_from_image(image_base64)

        print("Extracted data:")
        print("--------------------------------")
        print(extracted_data)
        print("--------------------------------")

        print("\n\n")

        print("Extracted data Dump:")
        print("--------------------------------")
        print(extracted_data.model_dump())
        print("--------------------------------")

    except ValueError as e:
        print(f"Error: {e}")
