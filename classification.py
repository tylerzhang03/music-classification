import random
random.seed(19340532)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt

music_raw = pd.read_csv('musicData.csv')

music = music_raw.copy()
mask = music['duration_ms'].isin([-1])
music.loc[mask, 'duration_ms'] = np.mean(music.loc[~mask, 'duration_ms'])
mask = music['tempo'].isin(['?'])
music.loc[mask, 'tempo'] = np.mean(music.loc[~mask, 'tempo'].astype(float))

key_to_num = {'A':1, 'A#':2, 'B':3, 'C':4, 'C#':5, 'D':6, 'D#':7, 'E':8, 'F':9, 'F#':10, 'G':11, 'G#':12}
music['key'] = music['key'].map(key_to_num)

music['mode'] = np.where(music['mode'] == 'Minor', 0, 1)

genre_to_num = {
    'Alternative':1,
    'Anime':2,
    'Blues':3,
    'Classical':4,
    'Country':5,
    'Electronic':6,
    'Hip-Hop':7,
    'Jazz':8,
    'Rap':9,
    'Rock':10}
music['music_genre'] = music['music_genre'].map(genre_to_num)

music = music.dropna().reset_index()
music = music.drop(columns=['index'])



scaler = StandardScaler()
music.iloc[:, np.r_[3:9,10:12,13:15,16]] = scaler.fit_transform(music.iloc[:, np.r_[3:9,10:12,13:15,16]])

embed = TSNE(n_components=2, perplexity=1000, init='pca', learning_rate='auto', n_jobs=-1)
music2d = embed.fit_transform(music.iloc[:, np.r_[3:15,16]])

embed3d = TSNE(n_components=3, perplexity=200, init='pca', learning_rate='auto', n_jobs=-1)
music3d = embed3d.fit_transform(music.iloc[:, np.r_[3:15,16]])
#music2d = embed.transform(music.iloc[:, np.r_[3:15,16]])

reducer = umap.UMAP()
umap_embed = reducer.fit_transform(music.iloc[:, np.r_[3:15,16]])

plt.scatter(umap_embed[:, 0], umap_embed[:, 1], c=music['music_genre'])
plt.title('UMAP 2d Embedding')
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.show()

plt.scatter(music2d[:, 0], music2d[:, 1], c=music['music_genre'])
plt.title('TSNE 2d Embedding')
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.show()



silhouettes = np.zeros(12)

for k in range(2, 12):
    kmeans = KMeans(k)
    labels = kmeans.fit_predict(music2d)
    silhouettes[k-1] = silhouette_score(music2d, labels)
    
plt.plot(np.arange(1, 13, 1), silhouettes, 'r-o', lw = 2)
plt.xlabel('Clusters k')
plt.ylabel('Silhouette Average')
plt.title('Elbow Curve for K-means')
plt.grid()
plt.show()

kmeans = KMeans(3)
labels = kmeans.fit_predict(music2d)

plt.scatter(music2d[:,0], music2d[:,1], c=labels)
plt.title('t-SNE 2d embedding with kmeans, k=3')
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.show()


silhouettes = np.zeros(12)

for k in range(2, 12):
    kmeans = KMeans(k)
    labels = kmeans.fit_predict(umap_embed)
    silhouettes[k-1] = silhouette_score(umap_embed, labels)
    
plt.plot(np.arange(1, 13, 1), silhouettes, 'r-o', lw = 2)
plt.xlabel('Clusters k')
plt.ylabel('Silhouette Average')
plt.title('Elbow Curve for K-means')
plt.grid()
plt.show()

kmeans = KMeans(2)
labels = kmeans.fit_predict(umap_embed)

plt.scatter(umap_embed[:,0], umap_embed[:,1], c=labels)
plt.title('UMAP 2d embedding with kmeans, k=2')
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.show()



embed_music = pd.DataFrame(music3d.copy())
embed_music.insert(3, 'music_genre', music['music_genre'], True)

test_set = embed_music.groupby('music_genre').sample(n=500)
training_set = embed_music.drop(test_set.index)

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(training_set.iloc[:, 0:3], training_set.iloc[:,3])
y_pred = clf.predict_proba(test_set.iloc[:,0:3])
y_true = test_set.iloc[:,3]
print(roc_auc_score(y_true, y_pred, multi_class='ovr'))



n_classes=10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.where(y_true == i+1, 1, 0), y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr_macro_avg = all_fpr
tpr_macro_avg = mean_tpr
roc_auc_macro = auc(fpr_macro_avg, tpr_macro_avg)

plt.figure()
lw = 2
colors = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(str({j for j in genre_to_num if genre_to_num[j] == i+1}).strip("{''}"), roc_auc[i]))

plt.plot(fpr_macro_avg, tpr_macro_avg,
         label='average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_macro),
         color='black', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()