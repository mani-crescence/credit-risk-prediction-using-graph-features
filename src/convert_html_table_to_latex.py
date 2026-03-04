import pandas as pd
from bs4 import BeautifulSoup

#  Function to convert HTML table to LaTeX
def html_table_to_latex(html_file, table_index=0):
    # Read the HTML file
    with open(html_file, 'r') as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all tables in the HTML content
    tables = soup.find_all('table')

    # Check if the specified table index exists
    if table_index >= len(tables):
        raise ValueError(f"Table index {table_index} out of range. Found {len(tables)} tables.")

    # Convert the specified table to a DataFrame
    df = pd.read_html(str(tables[table_index]))[0]

    # Convert DataFrame to LaTeX
    latex_table = df.to_latex(index=False)

    return latex_table

html_file = 'outputs/results/results/result_CAT_bondora.html'

# Get the LaTeX code for the first table (change the index if needed)
latex_code = html_table_to_latex(html_file, table_index=0)

# Print the LaTeX code
print(latex_code)
