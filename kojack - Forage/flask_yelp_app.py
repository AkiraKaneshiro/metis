from flask import Flask
from flask import request, render_template, jsonify

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix, vstack

import urlparse
import pymongo
import numpy as np



#---------- URLS AND WEB PAGES -------------#                                                                       

client = pymongo.MongoClient()

penn_sentiments_location = client.dsbc.penn_unwind_location


# Initialize the app                                                                                                
app = Flask(__name__)

@app.route("/")
def viz_page():
    """ Homepage: serve our Forage home page, yelp_app.html                                                         
    """

    return render_template("yelp_app.html")
    with open("templates/yelp_app.html", 'r') as viz_file:
        return viz_file.read()


@app.route('/yelp_result', methods=['GET', 'POST'])
def result():
    """ Search text inputted from homepage using MongoDB text search                                                
    """

# NEED TO MAKE DISLIKE BOX OPTIONAL. ONLY WORKS IF TEXT IS INPUTTED FOR BOTH BOXES                                  

    query = urlparse.parse_qs(request.get_data())
    like_text = query['like'][0]
    dislike_text = "-"+query['dislike'][0]
    text_search = like_text + " " + dislike_text


    # to make dislike box optional                                                                                  
    #if len(query['dislike'][0]) >0:                                                                               

    #else:                                                                                                          
    #    text_search = like_text                                                                                    

    text_search_result = penn_sentiments_location.find({"$text": {"$search": text_search}})
    #text_search_result = client.dsbc.command("text", "penn_sentiments_location", search=text_search)               


    print kmeans(text_search_result)
    #print toptfidf(text_search_result)                                                                             

    return render_template("yelp_result.html", text_search=text_search, text_search_result=text_search_result)


def kmeans(text_search_result):
    """ Cluster reviews and find review closest to centroid"""

    review_list, restaurant, sentiment = [], [], []
    for value in text_search_result:
	review_list.append(value['text'])
	restaurant.append(value['biz_name'])
        sentiment.append(value['polarity_score'])


    if not review_list:
        print 'TEXT SEARCH DID NOT FIND ANY RESULTS'

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,1))

    #yelp_vectors = vectorizer.fit(review_list)                                                                     
    review_vectors = vectorizer.fit_transform(review_list)

    n_clusters = 5
    #km = KMeans(n_clusters=k)
        km = MiniBatchKMeans(n_clusters=n_clusters)

    # Compute clustering and transform yelp_vectors to cluster-distance space                                       
    km_fit = km.fit_transform(review_vectors)

    sentence_clusters = zip(km.labels_,restaurant,review_list,review_vectors,sentiment)

#    print 'sentence_clusters', sentence_clusters                                                                   

    sentence_cluster_dict = {} # value: j[1]=restaurant_name, j[2]=review, j[3]=sparse matrix of tf-idf vectors     
                               # key: cluster_label                                                                 

    for sentence_cluster in sentence_clusters:
        cluster_label, review_data = sentence_cluster[0], sentence_cluster[1:]
        current_list = sentence_cluster_dict.setdefault(cluster_label, [])
        current_list.append(review_data)
        sentence_cluster_dict[cluster_label] = current_list

    cluster_dict = {}
    restaurant_cluster_weights = {}

    for cluster, review_data_list in sentence_cluster_dict.iteritems():


        center =  np.matrix(km.cluster_centers_[cluster])
        sentence_vectors = vstack([vector for (rest, sent, vector, sentiment) in review_data_list]) #vstack needed \
for sparse matrix                                                                                                   
        sentence_vectors = sentence_vectors.toarray()
        sentences = [sent for (rest, sent, vector, sentiment) in review_data_list]
        restaurant_names = [rest for (rest, sent, vector, sentiment) in review_data_list]
        sentiments = [sentiment for (rest, sent, vector, sentiment) in review_data_list]
        avr_sentiment = float(sum(sentiments))/len(sentiments)
        distances = pairwise_distances(sentence_vectors, center, metric='cosine')
        sentence_distances = sorted(list(zip(distances,sentences)))
        center_sentence = sentence_distances[0][1]
        similar_sentiment_count = len(sentences)
        close_sentences = [sent for dist, sent in sentence_distances[1:5]]
        for rest in list(set(restaurant_names)):
            current_cluster_weight = float(restaurant_names.count(rest)/len(restaurant_names))
            weights = restaurant_cluster_weights.setdefault(rest, {})
            weights[cluster] = current_cluster_weight

        print restaurant_cluster_weights
        # several weights are 0                                                                                     
       # find highest weight, return cluster number                                                                 


        #for i, j in restaurant_cluster_weights.iteritems():                                                        
        #    print i                                                                                                
        #    print j                                                                                                
        print 'keys', restaurant_cluster_weights.keys()

        cluster_dict[cluster] = {'center_sentence_text': center_sentence,
                                 'close_sentence_text': close_sentences,
                                 'cluster_sentence_count': similar_sentiment_count}
        #print cluster_dict                                                                                         




    return cluster_dict
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

