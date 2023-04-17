import requests

if __name__ == "__main__":
    url = "http://localhost:8765/translate_pdf/"
    input_pdf_path = "../assets/1706.03762.pdf"
    output_pdf_path = "../assets/translated_1706.03762.pdf"

    with open(input_pdf_path, "rb") as input_pdf:
        response = requests.post(url, files={"input_pdf": input_pdf})

    if response.status_code == 200:
        with open(output_pdf_path, "wb") as output_pdf:
            output_pdf.write(response.content)
        print(f"Converted PDF saved to {output_pdf_path}")
    else:
        print(f"An error occurred: {response.status_code}")
