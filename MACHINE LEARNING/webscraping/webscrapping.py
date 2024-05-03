import requests
from bs4 import BeautifulSoup
import pandas as pd


# Fetch HTML content
url = input('enter the url:')
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract data
data_list = []
for item in soup.find_all('img'):
      data = {
        'src': item.get('src'),
        'alt': item.get('alt'),
        # Add more fields as needed
      }
      data_list.append(data)

# Store data in Excel
df = pd.DataFrame(data_list)

urls=input("enter the filename:")
df.to_excel(urls, index=False)


