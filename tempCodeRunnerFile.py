ulSoup(raw_html, 'html.parser')
        total = 0
        recovered = 0
        deaths = 0
        total = BeautifulSoup.find(class="maincounter-number")
        print(total)