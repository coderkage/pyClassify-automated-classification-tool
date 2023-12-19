# Create timestamp for unique file name
# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# output = io.StringIO()
# with redirect_stdout(output):
#     cross_validation_and_evaluation(model, selected_features_data, crossval_method)

# # Save the output in a PDF file
# pdf = FPDF()
# pdf.add_page()
# pdf.set_font("Arial", size=12)
# pdf.cell(200, 10, txt="Cross-Validation and Evaluation Results", ln=True, align='C')
# pdf.ln(10)
# pdf.multi_cell(0, 10, txt=output.getvalue())
# pdf.output(f"report_{timestamp}.pdf")