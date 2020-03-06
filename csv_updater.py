# Initial Data is provided using John Hopkin's dataset (Time Series)

# The Dataset needs to be constantly updated using the scraper.

# Maincounter Number
# Structure

# Skipping the number of suspected cases because of lack of data
# The deaths and infected are stored in the string and simple slicing will get us the new values


from bs4 import BeautifulSoup
from urllib import request
def get_nums():
    url = 'https://www.worldometers.info/coronavirus/'
    html = request.urlopen(url).read().decode('utf8')
    html[:60]
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title')
    total_cases = title.string[27:33]
    deaths = title.string[44:50]
    total_cases = int(total_cases.replace(",", ''))
    deaths = int(deaths.replace(",", ''))
    return total_cases, deaths

get_nums()
