from reportlab.pdfgen import canvas

def create_pdf(file_name):
    c = canvas.Canvas(file_name)
    c.drawString(100, 750, "Sample Text for Printing Test")
    c.drawString(100, 700, "Second line of text.")
    c.showPage()
    c.save()

create_pdf("sample.pdf")

