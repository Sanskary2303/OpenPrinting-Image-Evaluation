#!/usr/bin/env python3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image
import os
import math

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

def create_multipage_document(filename, pages=5, reorder=False):
    """Create a PDF with multiple pages with page numbers and markers for testing page order
    
    Parameters:
    - filename: Output PDF file path
    - pages: Number of pages to generate
    - reorder: If True, reorder pages in a predictable way to test detection
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import math
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    page_order = list(range(pages))
    if reorder and pages > 2:
        page_order[1], page_order[2] = page_order[2], page_order[1]
        
        if pages > 3:
            last_idx = pages - 1
            middle_idx = pages // 2
            page_order[last_idx], page_order[middle_idx] = page_order[middle_idx], page_order[last_idx]
    
    for i in range(pages):
        page_idx = page_order[i]
        
        c.setFont('Helvetica-Bold', 36)
        c.drawString(72, height - 72, f"Page {page_idx + 1} of {pages}")
        
        c.setFont('Helvetica', 10)
        c.drawString(36, height - 36, f"P{page_idx + 1}")  # Top left
        c.drawString(width - 72, height - 36, f"P{page_idx + 1}")  # Top right
        c.drawString(36, 36, f"P{page_idx + 1}")  # Bottom left
        c.drawString(width - 72, 36, f"P{page_idx + 1}")  # Bottom right
        
        c.setFillColorRGB(0.1 * page_idx, 0.1 * ((pages - page_idx) % pages), 0.5)
        
        shape_size = 2 * inch
        center_x = width / 2
        center_y = height / 2
        
        if page_idx % 5 == 0:
            c.circle(center_x, center_y, shape_size/2, fill=1)
        elif page_idx % 5 == 1:
            c.rect(center_x - shape_size/2, center_y - shape_size/2, 
                  shape_size, shape_size, fill=1)
        elif page_idx % 5 == 2:
            c.setFillColorRGB(0.1 * page_idx, 0.1 * ((pages - page_idx) % pages), 0.5)
            
            p = c.beginPath()
            p.moveTo(center_x, center_y + shape_size/2)
            p.lineTo(center_x - shape_size/2, center_y - shape_size/2)
            p.lineTo(center_x + shape_size/2, center_y - shape_size/2)
            p.close()
            c.drawPath(p, fill=1)
            
        elif page_idx % 5 == 3:
            points = 5
            c.setFillColorRGB(0.1 * page_idx, 0.1 * ((pages - page_idx) % pages), 0.5)
            radius_outer = shape_size/2
            radius_inner = radius_outer * 0.4
            angle = math.pi / points
            
            p = c.beginPath()
            for i in range(2*points):
                radius = radius_outer if i % 2 == 0 else radius_inner
                x = center_x + radius * math.sin(i * angle)
                y = center_y + radius * math.cos(i * angle)
                if i == 0:
                    p.moveTo(x, y)
                else:
                    p.lineTo(x, y)
            p.close()
            c.drawPath(p, fill=1)
            
        else:
            c.setFillColorRGB(0.1 * page_idx, 0.1 * ((pages - page_idx) % pages), 0.5)
            
            p = c.beginPath()
            p.moveTo(center_x, center_y + shape_size/2)
            p.lineTo(center_x + shape_size/2, center_y)
            p.lineTo(center_x, center_y - shape_size/2)
            p.lineTo(center_x - shape_size/2, center_y)
            p.close()
            c.drawPath(p, fill=1)
        
        c.setFillColorRGB(1, 1, 1)
        c.setFont('Helvetica-Bold', 72)
        text_width = c.stringWidth(str(page_idx + 1), 'Helvetica-Bold', 72)
        c.drawString(center_x - text_width/2, center_y - 24, str(page_idx + 1))
        
        bar_width = width * (page_idx + 1) / (pages * 2)
        c.setFillColorRGB(0, 0, 0)
        c.rect(72, 72, bar_width, 20, fill=1)
        
        # QR code like pattern unique to page number for easier detection
        cell_size = 10
        grid_dim = 5
        x_offset = width - 72 - (grid_dim * cell_size)
        y_offset = 72
        
        for gx in range(grid_dim):
            for gy in range(grid_dim):
                if ((gx + gy * grid_dim) + page_idx) % 2 == 0:
                    c.setFillColorRGB(0, 0, 0)
                    c.rect(x_offset + gx*cell_size, 
                          y_offset + gy*cell_size, 
                          cell_size, cell_size, fill=1)
        
        c.showPage()
    
    c.save()
    
    print(f"Created {pages}-page document at {filename}")
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

def create_test_pdf(output_path):
    """Create a test PDF document with various elements"""
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    
    # Add title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(inch, height - inch, "CUPS Filter Test Document")
    
    # Add some text
    c.setFont("Helvetica", 12)
    c.drawString(inch, height - 2*inch, "This is a test document to validate CUPS filter chains.")
    c.drawString(inch, height - 2.5*inch, "It contains text, graphics, and various elements.")
    
    # Add a rectangle
    c.setStrokeColor(colors.blue)
    c.setFillColor(colors.lightblue)
    c.rect(inch, height - 4*inch, 2*inch, inch, fill=1)
    
    # Add a circle
    c.setStrokeColor(colors.red)
    c.setFillColor(colors.pink)
    c.circle(4*inch, height - 3.5*inch, 0.5*inch, fill=1)
    
    # Add page number
    c.setFont("Helvetica", 10)
    c.drawString(width/2, 0.5*inch, "Page 1")
    
    # Save the PDF
    c.save()

def create_text_file(output_path):
    """Create a test text file"""
    with open(output_path, 'w') as f:
        f.write("CUPS Filter Test Document\n")
        f.write("=========================\n\n")
        f.write("This is a simple text file to test CUPS text filters.\n")
        f.write("It contains multiple lines of plain text.\n\n")
        f.write("The text/plain MIME type is typically processed by texttopdf.\n")
        f.write("This filter converts text to PDF before further processing.\n")

def create_postscript_file(output_path):
    """Create a simple PostScript file"""
    ps_content = """%!PS-Adobe-3.0
%%Title: CUPS Filter Test Document
%%Creator: generate_test_documents.py
%%Pages: 1
%%DocumentData: Clean7Bit
%%Orientation: Portrait
%%PageOrder: Ascend
%%BoundingBox: 0 0 596 842
%%EndComments
%%BeginProlog
%%EndProlog
%%Page: 1 1
/Helvetica findfont 24 scalefont setfont
72 770 moveto
(CUPS Filter Test Document) show
/Helvetica findfont 12 scalefont setfont
72 720 moveto
(This is a test PostScript document.) show
72 700 moveto
(It will be processed by PostScript filters in CUPS.) show
stroke
showpage
%%EOF
"""
    with open(output_path, 'w') as f:
        f.write(ps_content)

def create_test_documents(output_dir):
    """Create test documents of various MIME types"""
    # Create PDF
    create_test_pdf(os.path.join(output_dir, "test.pdf"))
    
    # Create text file
    create_text_file(os.path.join(output_dir, "test.txt"))
    
    # Create PostScript file
    create_postscript_file(os.path.join(output_dir, "test.ps"))
    
    # For JPEG, we would typically need an image library
    # Here we're just noting that you might want to add one
    print("Note: Add a sample JPEG file manually to test image/jpeg filters")

if __name__ == "__main__":
    # test_files = generate_all_test_documents()
    # print(f"Generated {len(test_files)} test documents:")
    # for file in test_files:
    #     print(f"  - {file}")
    output_dir = "test_files"
    os.makedirs(output_dir, exist_ok=True)
    create_test_documents(output_dir)
    print(f"Test documents created in {output_dir}/")