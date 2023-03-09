import numpy as np
import pandas as pd

df = pd.read_csv("imdb_reviews.csv")
unique_movies = df['movie_name'].unique()

print("Number of movies: ", len(unique_movies))
print("Unique movies: ")
print(unique_movies)




# output =>

# Number of movies:  37

# Unique movies:
# ['Inception (2010)' 'Ant-Man and the Wasp: Quantumania (2023)'
#  'Cocaine Bear (2023)' 'The Whale (2022)' 'Babylon (2022)'
#  'Knock at the Cabin (2023)' 'Sharper (2023)'
#  'The Banshees of Inisherin (2022)'
#  'Winnie the Pooh: Blood and Honey (2023)'
#  'Avatar: The Way of Water (2022)' 'Black Panther: Wakanda Forever (2022)'
#  'Everything Everywhere All at Once (2022)'
#  'Im Westen nichts Neues (2022)' 'Infinity Pool (2023)'
#  'Your Place or Mine (2023)' 'The Menu (2022)'
#  'Puss in Boots: The Last Wish (2022)' 'M3GAN (2022)'
#  'Bullet Train (2022)' 'The Fabelmans (2022)' 'The Woman King (2022)'
#  'The Strays (2023)' 'Tár (2022)' 'Women Talking (2022)'
#  'Triangle of Sadness (2022)' 'We Have a Ghost (2023)'
#  'Somebody I Used to Know (2023)' 'White Noise (2022)'
#  'Top Gun: Maverick (2022)' 'Plane (2023)' 'Aftersun (2022)'
#  'Jesus Revolution (2023)' 'Glass Onion (2022)' 'You People (2023)'
#  'Titanic (1997)' "Magic Mike's Last Dance (2023)" 'Ant-Man (2015)']