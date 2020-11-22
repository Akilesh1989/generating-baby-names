import requests
import csv
from bs4 import BeautifulSoup


url = "https://www.valaitamil.com/girl-baby-names/A"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
href_elements = soup.find_all("a", href=True)
names = []
for a in href_elements:
  if a['href'].endswith('html') and 'font-size:11px' in str(a):
    names.append(a.text.strip().lower())

names_2 = []
unique_names = 0
for i in range(2, 300):
  if unique_names <= 1000:
    next_url = f"https://www.valaitamil.com/girl_baby_names/A/page-{i}"
    print(next_url)
    page = requests.get(next_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    href_elements = soup.find_all("a", href=True)
    for a in href_elements:
      if a['href'].endswith('html') and 'font-size:11px' in str(a):
        names_2.append(a.text.strip().lower())
    unique_names = len(list(set(names_2)))
  else:
    break

all_names = names + names_2
unique_names = list(set(all_names))
unique_names = list(map(lambda s: s + '.', unique_names))


with open("names.csv", 'w', newline='') as csv_file:
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL, delimiter='\n')
    wr.writerow(unique_names)