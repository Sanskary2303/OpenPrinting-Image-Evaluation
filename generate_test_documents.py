from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image
import os
import random

FONTS = ['Helvetica', 'Courier', 'Times-Roman']
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    FONTS.append('DejaVuSans')
except:
    print("DejaVuSans font not available, using standard fonts only")

try:
    pdfmetrics.registerFont(TTFont('FreeSans', '/usr/share/fonts/truetype/freefont/FreeSans.ttf'))
    FONTS.append('FreeSans')
except:
    print("FreeSans font not available, using standard fonts only")

try:
    pdfmetrics.registerFont(TTFont('Helvetica-Italic', '/usr/share/fonts/truetype/helvetica/Italic.ttf'))
except:
    print("Helvetica-Italic font not registered, using standard fonts for italic text")

SAMPLE_TEXTS = {
    'english': "The quick brown fox jumps over the lazy dog.",
    'spanish': "El veloz zorro marrón salta sobre el perro perezoso.",
    'french': "Le rapide renard brun saute par-dessus le chien paresseux.",
    'german': "Der schnelle braune Fuchs springt über den faulen Hund.",
    'chinese': "快速的棕色狐狸跳过懒狗。",
    'russian': "Быстрая коричневая лиса прыгает через ленивую собаку.",
    'arabic': "الثعلب البني السريع يقفز فوق الكلب الكسول."
}

FONT_SIZES = [8, 10, 12, 14, 16, 18, 24, 36]

def create_text_pdf(filename, language='english', font='Helvetica', font_size=12):
    """Create a PDF with text in specified language and font"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont(font, font_size)
    
    c.setFont(font, font_size + 4)
    c.drawString(72, height - 72, f"Text Test Document - {language}")
    
    y_position = height - 100
    text = SAMPLE_TEXTS.get(language, SAMPLE_TEXTS['english'])
    
    c.setFont(font, font_size)
    c.drawString(72, y_position, text)
    y_position -= 30
    
    try:
        bold_font = font + '-Bold'
        c.setFont(bold_font, font_size)
        c.drawString(72, y_position, f"Bold: {text}")
        y_position -= 30
    except:
        pass
    
    try:
        italic_font = font + '-Italic'
        c.setFont(italic_font, font_size)
        c.drawString(72, y_position, f"Italic: {text}")
        y_position -= 30
    except:
        pass
    
    for size in [8, 12, 18, 24]:
        if size != font_size:
            c.setFont(font, size)
            c.drawString(72, y_position, f"{size}pt: {text}")
            y_position -= size + 10
    
    y_position -= 30
    c.setFont(font, font_size)
    
    c.drawString(72, y_position, "Left aligned: " + text)
    y_position -= 30
    
    c.drawCentredString(width/2, y_position, "Center aligned: " + text)
    y_position -= 30
    
    c.drawRightString(width - 72, y_position, "Right aligned: " + text)
    
    c.showPage()
    c.save()
    return filename

def create_image_pdf(filename):
    """Create a PDF with images"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont('Helvetica', 16)
    c.drawString(72, height - 72, "Image Test Document")
    
    y_position = height - 100
    
    c.setStrokeColor(colors.blue)
    c.setFillColor(colors.lightblue)
    c.rect(72, y_position - 50, 200, 50, fill=True)
    y_position -= 70
    
    c.setStrokeColor(colors.red)
    c.setFillColor(colors.pink)
    c.circle(172, y_position - 50, 50, fill=True)
    y_position -= 120
    
    images = ['test_image1.jpg', 'test_image2.png', 'test_image3.jpg']
    for img_file in images:
        if os.path.exists(img_file):
            img = Image(img_file, 3*inch, 2*inch)
            img.drawOn(c, 72, y_position - 2*inch)
            y_position -= 2.5*inch
    
    for i in range(10):
        c.setFillColorRGB(i/10, 0.5, 1-i/10)
        c.rect(72 + i*40, 100, 40, 40, fill=True)
    
    c.showPage()
    c.save()
    return filename

def create_mixed_pdf(filename):
    """Create a PDF with mixed content (text and images)"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont('Helvetica', 16)
    c.drawString(72, height - 72, "Mixed Content Test Document")
    
    y_position = height - 100
    
    c.setFont('Helvetica', 12)
    c.drawString(72, y_position, "This document contains both text and graphical elements.")
    y_position -= 30
    
    c.setStrokeColor(colors.darkgreen)
    c.setFillColor(colors.lightgreen)
    c.rect(72, y_position - 60, 400, 50, fill=True)
    
    c.setFillColor(colors.black)
    c.setFont('Helvetica', 14)
    c.drawString(100, y_position - 30, "Text inside a colored box")
    y_position -= 80
    
    c.setStrokeColor(colors.black)
    c.line(72, y_position, 500, y_position)
    y_position -= 30
    
    c.setFont('Helvetica', 12)
    lorem_ipsum = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Suspendisse euismod dolor a augue fringilla, in faucibus ante dapibus. 
    Vivamus quis vehicula felis. Morbi pharetra felis a libero ultricies condimentum."""
    
    text_object = c.beginText(72, y_position)
    text_object.setFont('Helvetica', 12)
    
    for line in lorem_ipsum.split('\n'):
        text_object.textLine(line.strip())
    
    c.drawText(text_object)
    y_position -= 100
    
    c.setStrokeColor(colors.red)
    c.setFillColor(colors.pink)
    c.circle(150, y_position, 40, fill=True)
    
    c.setFillColor(colors.black)
    c.drawString(200, y_position, "Text next to a circle")
    
    c.showPage()
    c.save()
    return filename

def create_complex_layout_pdf(filename):
    """Create a PDF with complex layout (multiple columns, tables, etc.)"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont('Helvetica-Bold', 18)
    c.drawString(72, height - 72, "Complex Layout Test Document")
    
    c.setStrokeColor(colors.black)
    c.line(72, height - 90, width - 72, height - 90)
    
    col_width = (width - 144) / 2
    left_col_x = 72
    right_col_x = 72 + col_width + 20
    
    y_position = height - 120
    
    c.setFont('Helvetica-Bold', 14)
    c.drawString(left_col_x, y_position, "Column 1")
    
    c.drawString(right_col_x, y_position, "Column 2")
    y_position -= 30
    
    c.setFont('Helvetica', 10)
    left_text = """This is the left column with some content.
    It demonstrates multi-column layout capabilities.
    This can be used to test how well the print pipeline
    preserves complex layouts and positioning."""
    
    text_object = c.beginText(left_col_x, y_position)
    text_object.setFont('Helvetica', 10)
    
    for line in left_text.split('\n'):
        text_object.textLine(line.strip())
    
    c.drawText(text_object)
    
    c.setFont('Courier', 10)
    right_text = """This is the right column with different font.
    Testing how well different fonts within the same
    document are preserved after processing through
    the print pipeline."""
    
    text_object = c.beginText(right_col_x, y_position)
    text_object.setFont('Courier', 10)
    
    for line in right_text.split('\n'):
        text_object.textLine(line.strip())
    
    c.drawText(text_object)
    
    y_position -= 120
    table_width = width - 144
    col_widths = [table_width * 0.3, table_width * 0.4, table_width * 0.3]
    
    c.setFillColor(colors.lightgrey)
    c.rect(left_col_x, y_position - 20, table_width, 20, fill=True)
    
    c.setFillColor(colors.black)
    c.setFont('Helvetica-Bold', 10)
    x_pos = left_col_x
    
    headers = ["Column A", "Column B", "Column C"]
    for i, header in enumerate(headers):
        c.drawString(x_pos + 5, y_position - 15, header)
        x_pos += col_widths[i]
    
    y_position -= 20
    
    rows = [
        ["Row 1, Cell 1", "Row 1, Cell 2", "Row 1, Cell 3"],
        ["Row 2, Cell 1", "Row 2, Cell 2", "Row 2, Cell 3"],
        ["Row 3, Cell 1", "Row 3, Cell 2", "Row 3, Cell 3"]
    ]
    
    c.setFont('Helvetica', 10)
    for row in rows:
        x_pos = left_col_x
        
        if rows.index(row) % 2 == 1:
            c.setFillColor(colors.lightgrey)
            c.rect(left_col_x, y_position - 20, table_width, 20, fill=True)
        
        c.setFillColor(colors.black)
        for i, cell in enumerate(row):
            c.drawString(x_pos + 5, y_position - 15, cell)
            x_pos += col_widths[i]
        
        y_position -= 20
    
    try:
        c.setFont('Helvetica-Italic', 9)
    except:
        c.setFont('Helvetica', 9)
        print("Using regular Helvetica for footer (italic not available)")
    
    c.drawCentredString(width/2, 30, "Test document generated for print pipeline testing")
    c.drawCentredString(width/2, 20, "Page 1")
    
    c.showPage()
    c.save()
    return filename

def generate_all_test_documents(output_dir='.'):
    """Generate a full set of test documents with different properties"""
    test_files = []
    
    for lang in ['english', 'spanish', 'french', 'german']:
        filename = os.path.join(output_dir, f"text_{lang}.pdf")
        create_text_pdf(filename, language=lang)
        test_files.append(filename)
    
    for font in FONTS:
        filename = os.path.join(output_dir, f"text_font_{font}.pdf")
        create_text_pdf(filename, font=font)
        test_files.append(filename)
    
    for size in [8, 12, 18, 24]:
        filename = os.path.join(output_dir, f"text_size_{size}.pdf")
        create_text_pdf(filename, font_size=size)
        test_files.append(filename)
    
    filename = os.path.join(output_dir, "image_document.pdf")
    create_image_pdf(filename)
    test_files.append(filename)
    
    filename = os.path.join(output_dir, "mixed_content.pdf")
    create_mixed_pdf(filename)
    test_files.append(filename)
    
    filename = os.path.join(output_dir, "complex_layout.pdf")
    create_complex_layout_pdf(filename)
    test_files.append(filename)
    
    return test_files

if __name__ == "__main__":
    test_files = generate_all_test_documents()
    print(f"Generated {len(test_files)} test documents:")
    for file in test_files:
        print(f"  - {file}")