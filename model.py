import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split   
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle



def classify_genre(genre):
    electronic_genres = ['electronic', 'edm', 'techno', 'dubstep', 'house', 'detroit-techno', 'minimal-techno', 'trance', 'chicago-house', 'deep-house','electro']
    folk_genres = ['folk', 'bluegrass', 'country', 'acoustic', 'singer-songwriter', 'indie-folk','classical']
    funk_soul_genres = ['funk', 'soul', 'r-n-b', 'gospel']
    hip_hop_genres = ['hip-hop', 'rap', 'breakbeat','trip-hop']
    jazz_genres = ['jazz', 'swing', 'bebop']
    latin_genres = ['latin', 'salsa', 'samba', 'reggaeton']
    pop_genres = ['pop', 'pop-film', 'power-pop', 'indie-pop','synth-pop','k-pop','j-pop']
    reggae_genres = ['reggae', 'dub']
    rock_genres = ['rock', 'alt-rock', 'hard-rock', 'punk-rock', 'grunge', 'rock-n-roll', 'metal', 'heavy-metal', 'progressive-house', 'punk', 'hardcore', 'industrial', 'hardstyle', 'grindcore', 'death-metal', 'metalcore','black-metal','rockabilly','psych-rock','j-rock']
    stage_screen_genres = ['anime', 'children', 'comedy', 'disney', 'show-tunes']


    genre = genre.lower()

    if genre in electronic_genres:
        return 1
    elif genre in folk_genres:
        return 2
    elif genre in funk_soul_genres:
        return 3
    elif genre in hip_hop_genres:
        return 4
    elif genre in jazz_genres:
        return 5
    elif genre in latin_genres:
        return 6
    elif genre in pop_genres:
        return 7
    elif genre in reggae_genres:
        return 8
    elif genre in rock_genres:
        return 9
    elif genre in stage_screen_genres:
        return 10
    else:
        return 11  # You can customize this as needed




file_object = pd.read_csv('train.csv')


y = file_object.track_genre 
features = ['popularity','danceability' , 'energy'  , 'loudness' , 'acousticness' , 'instrumentalness' ,'tempo','duration_ms','key','liveness','mode','time_signature']

X=file_object[features]
distinct_elements = file_object.track_genre.drop_duplicates().tolist()
y = y.tolist()

for i in y:
    y[y.index(i)] = classify_genre(i)


X_train , X_test , Y_train , Y_test = train_test_split(X,y,test_size = 0.4 ,train_size = 0.6 ,random_state = 41)

X_test_top = []
Y_test_top = []

for i in range(len(Y_test)):
    if Y_test[i] in [9,1,2,7,3]:
        pass
    else:
        Y_test_top.append(Y_test[i])
        X_test_top.append(X_test.iloc[i])



scaler=StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test_top = scaler.transform(X_test_top)

k = 3
pca = PCA(n_components=k)

pca.fit(X_train)
X_train = pca.transform(X_train)
X_test_top = pca.transform(X_test_top)
svmc = LinearSVC(penalty='l2',dual=False)
svmc.fit(X_train,Y_train)



with open("model.pkl", 'rb') as file:  
    svmc = pickle.load(file)

predictions = svmc.predict(X_test_top)

print(predictions)



