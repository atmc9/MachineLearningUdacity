# import libraries
import urllib
import re
import sys
import csv
import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob


urls = [('amazon', 'https://www.amazon.com/s/ref=nb_sb_noss_2?%s')
        ]

def amazon_get_product_info(product_name):
        # get amazon url
        amazon_info = [(site,url) for (site,url) in urls if site == 'amazon']
        site, url = amazon_info[0]

        # query the website and return the html to 'page_data'
        params = urllib.parse.urlencode({'url': 'search-alias=aps', 'field-keywords': product_name})
        url = url % params
        # with open('amazondata.txt', 'r') as f:
        #     page_data = f.read()
        #     f.closed
        try:
            page_data = urllib.request.urlopen(url).read().decode("utf-8")
            with open('amazondata.txt', 'w') as f:
                 f.write(page_data)
                 f.closed
        except Exception as ex:
            print('exception: ' , ex)
            page_data = ''
            with open('amazondata.txt', 'r') as f:
                page_data = f.read()
                f.closed
        soup = BeautifulSoup(page_data, features="html.parser")
        result = []
        i = 0
        # loop through each row
        for row in soup.find_all('li', attrs={'id': re.compile('result_(\d+)')}):
            i = i+1

            # initialize to default values
            index = i
            title = ''
            link = ''
            price = 0
            rating_value = ''
            rating_member_count = 0

            name = row.find('a', attrs={'class': 'a-link-normal s-access-detail-page s-color-twister-title-link a-text-normal'})
            link = name['href']
            title_h2 = name.find('h2')
            title = title_h2.text if title_h2 is not None else product_name

            # get price whole
            price_whole = row.find('span', attrs={'class': 'sx-price-whole'})
            if price_whole is not None:
                price = int(price_whole.text.replace(',',''))

            # get fractional price
            price_fractional = row.find('span', attrs={'class': 'sx-price-fractional'})
            if price_fractional is not None:
                price = price + float('0.' + price_fractional.text)


            rating_div = row.find('div', attrs={'class': 'a-column a-span5 a-span-last'})

            if rating_div is not None:
                # get rating value
                rating = rating_div.find('span', attrs={'class': 'a-icon-alt'})
                if rating is not None:
                    rating_value = rating.text

                # get rating member count
                rating_count = rating_div.find('a', attrs={'class': 'a-size-small a-link-normal a-text-normal'})
                if rating_count is not None:
                    rating_member_count = int(rating_count.text.replace(',',''))

            result.append((index, title, link, price, rating_value, rating_member_count))

        print(result)
        save_file(result)
        return result

def save_file(result):
    with open('amazon.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['index', 'title', 'link', 'price', 'rating_value', 'rating_member_count', 'datetime'])
        for tuple in result:
            writer.writerow([tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5], datetime.datetime.now()])

amazon_get_product_info('iPhone X')
