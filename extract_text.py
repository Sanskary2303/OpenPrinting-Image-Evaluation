import pytesseract
from PIL import Image

img = Image.open('sample.png')

text = pytesseract.image_to_string(img)
print("Extracted Text:\n", text)

