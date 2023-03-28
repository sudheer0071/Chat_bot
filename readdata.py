
import pandas as pd
import requests


url = 'https://www.equitymaster.com/stockquotes/0-ADANI/list-of-adani-group-stocks'
gt = pd.read_html(url)


print(gt)