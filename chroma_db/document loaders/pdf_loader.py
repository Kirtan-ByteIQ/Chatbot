import os
import fitz               
import pytesseract
import ollama
from io import BytesIO
from PIL import Image
from langchain_core.documents import Document
def describe_image_with_moondream(image_bytes):
    try:
        response = ollama.chat(
            model="moondream",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Describe this image in detail. If it is a chart, "
                        "graph or table, explain what data/trend it is showing."
                    ),
                    "images": [image_bytes],
                }
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"error:{e}")
        return ""


def load_pdf_with_images(pdf_path):
    print(f"Reading: {pdf_path}")
    docs = []
    pdf = fitz.open(pdf_path)

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        page_text = page.get_text()
        if page_text.strip():
            docs.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_number + 1,
                        "type": "text",
                    },
                )
            )
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                pil_image = Image.open(BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(pil_image).strip()
            except Exception:
                ocr_text = ""

            caption = describe_image_with_moondream(image_bytes)

            combined = f"{caption}\n\nText found inside image: {ocr_text}".strip()

            if combined:
                docs.append(
                    Document(
                        page_content=combined,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_number + 1,
                            "type": "image",
                            "image_index": img_index,
                        },
                    ))

    pdf.close()
    print(f"   -> {len(docs)} chunks extracted (text + images)")
    return docs


def load_all_pdfs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            all_docs.extend(load_pdf_with_images(full_path))
    return all_docs
