

import pandas as pd

javascriptdata = pd.read_json("intent.json")


dataset = pd.DataFrame(javascriptdata)

print(dataset)

dataset.to_csv("box_data.csv")


print("This is the data present in the csv file : \n")

csv_fil = pd.read_csv("box_data.csv")








