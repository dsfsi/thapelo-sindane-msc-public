from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.manifold import TSNE
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def align_lex(file_source, file_target, bilex, path_out,l1, l2,embed_type = 'cca'):

    # if embeddings file has a header, then need to exclude it
    if embed_type == 'cca': #no header
        header = True
    elif embed_type == 'vecmap': #has a header
        header = False
    else:
        header = True
        
    vecs = KeyedVectors.load_word2vec_format(file_source, binary=False, no_header=header)
    vectar = KeyedVectors.load_word2vec_format(file_target, binary=False, no_header=header)
    
    name1=l1+'2'+l2+'_'+l1+'vec.txt'
    name2=l1+'2'+l2+'_'+l2+'vec.txt'

    name1 = os.path.join(path_out,name1)
    name2 = os.path.join(path_out,name2)
    
    with open(name1, 'wb') as fp:
        pickle.dump(vecs, fp)
    with open(name2, 'wb') as fp:
        pickle.dump(vectar, fp)
        
    file = open(bilex, "r")
    contents = file.readlines()
    
    source=[]
    target=[]
    for line in contents:
        line = line.replace('\n','')
        bi = line.split('|||')
        mark = len(bi[0])
        if bi[0][mark-1] == ' ':
            bi[0] = bi[0][0:mark-1]

        mark1 = len(bi[1])
        if bi[1][mark1-1] == ' ':
            bi[1] = bi[1][0:mark1-1]
        if bi[1][0] == ' ':
            bi[1] = bi[1][1:len(bi[1])]

        dup = source.count(bi[0])
        if dup == 0:
            source.append(bi[0])
            target.append(bi[1])
    
    d1 = vecs.key_to_index
    d1 = list(d1.keys())
    d2 = vectar.key_to_index
    d2 = list(d2.keys())

    source1=[]
    target1=[]
    for l in range(0,len(source)):
        cnt1 = 0
        cnt2 = 0
        words = source[l].split(' ')
        numwords = len(source[l].split(' '))
        for n in words:
            if n in d1:
                cnt1 += 1
        words1 = target[l].split(' ')
        numwords1 = len(target[l].split(' '))
        for nn in words1:
            if nn in d2:
                cnt2 += 1
        if (numwords == cnt1) and (numwords1 == cnt2):
            source1.append(source[l])
            target1.append(target[l])
    
    name1=l1+'2'+l2+'_'+l1+'.txt'
    name1 = os.path.join(path_out,name1)
    name2=l1+'2'+l2+'_'+l2+'.txt'
    name2 = os.path.join(path_out,name2)
    
    with open(name1, 'wb') as fp:
        pickle.dump(source1, fp)
    with open(name2, 'wb') as fp:
        pickle.dump(target1, fp)
            

def plot_bilex(source1, target1, vecs, vectar, fig_name, start=0, end=500):
    
    #vecs = KeyedVectors.load_word2vec_format(file_source, binary=False, no_header=True)
    #vectar = KeyedVectors.load_word2vec_format(file_target, binary=False, no_header=True)
    
    keys1 = source1
    keys2 = target1
    
    def avg_vec(word1,vecs):
        v=np.empty(200)
        cnt = 0
        words = word1.split(' ')
        numwords = len(word1.split(' '))
        for w in words:
            cnt += 1
            v = v+vecs[w]

        new_vec = v / cnt

        return new_vec
    
    def cosine_similarity(a, b):
        nominator = np.dot(a, b)

        a_norm = np.sqrt(np.sum(a**2))
        b_norm = np.sqrt(np.sum(b**2))

        denominator = a_norm * b_norm

        cosine_similarity = nominator / denominator
    
        return cosine_similarity
    
    embedding_clusters = []
    word_clusters = []
    cosine = []
    for word1,word2 in zip(keys1,keys2):
        embeddings = []
        words = []
        words.append(word1)
        ww1 = word1.split(' ')
        vec1 = avg_vec(word1,vecs)
        #embeddings.append(vecs[ww1[0]])
        embeddings.append(vec1)
        words.append(word2)
        ww2 = word2.split(' ')
        vec2 = avg_vec(word2,vectar)
        cos = cosine_similarity(vec1,vec2)
        cosine.append([word1,word2,cos])
        #embeddings.append(vectar[ww2[0]])
        embeddings.append(vec2)
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    ec_all = embedding_clusters 
    embedding_clusters = embedding_clusters[start:end]
    wc_all = word_clusters
    word_clusters = word_clusters[start:end]
        
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    
    def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
        plt.figure(figsize=(16, 9))
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
            x = embeddings[:, 0]
            y = embeddings[:, 1]
            color=['red','black']
            plt.scatter(x, y, c=color, alpha=a, label=label)
            for i, word in enumerate(words):
                plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom', size=8)
        #plt.legend(loc=4)
        plt.title(title)
        plt.grid(True)
        #plt.xlim([0,10])
        #plt.ylim([-10,10])
        if filename:
            plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        plt.show()
    
    tsne_plot_similar_words(('Similar words: '+fig_name), keys1, embeddings_en_2d, word_clusters, 0.7,
                        fig_name)
    
    return keys1, ec_all, wc_all, cosine

def tsne_plot(title, labels, embedding_clusters, word_clusters, a, start, end, filename=None):

    embedding_clusters = embedding_clusters[start:end]
    word_clusters = word_clusters[start:end]
    
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        color=['red','black']
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    #plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    #plt.xlim([0,10])
    #plt.ylim([-10,10])
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

def pca_bilex(source1, target1, file_source, file_target, start=0, end=500, xlim=None, ylim=None):
    
    worded = source1[start:end]
    worded1 = target1[start:end]

    vecs = KeyedVectors.load_word2vec_format(file_source, binary=False, no_header=True)
    vectar = KeyedVectors.load_word2vec_format(file_target, binary=False, no_header=True)
    
    def avg_vec(word1,vecs):
        v=np.empty(200)
        cnt = 0
        words = word1.split(' ')
        numwords = len(word1.split(' '))
        for w in words:
            cnt += 1
            v = v+vecs[w]

        new_vec = v / cnt

        return new_vec
    
    subset=[]
    for w in worded:
        #print(w)
        vec = avg_vec(w,vecs)
        subset.append(vec)

    subset1=[]
    for w in worded1:
        vec = avg_vec(w,vectar)
        subset1.append(vec)

    model = PCA(n_components=2)
    reduced = model.fit(subset).transform(subset)
    reduced = pd.DataFrame(reduced, columns="X Y".split())
    reduced["Word"] = worded

    model1 = PCA(n_components=2)
    reduced1 = model1.fit(subset1).transform(subset1)
    reduced1 = pd.DataFrame(reduced1, columns="X Y".split())
    reduced1["Word"] = worded1

    x=reduced['X']
    y=reduced['Y']
    x1=reduced1['X']
    y1=reduced1['Y']
    
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y, c='black', alpha=0.7, label=worded)
    for i, word in enumerate(worded):
        plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom', size=8)
    plt.scatter(x1, y1, c='red', alpha=0.7, label=worded1)
    for i, word in enumerate(worded1):
        plt.annotate(word, alpha=0.5, xy=(x1[i], y1[i]), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom', size=8)

    if xlim != None:
        plt.xlim(xlim)
        plt.ylim(ylim)





