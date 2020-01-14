import pandas as pd
import numpy as np
import os
import json
import spacy
import string as str_lib

#sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#OCR/image recognition
from PIL import Image
import pytesseract
import face_recognition

#for scraping
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import re
import time
import urllib

nlp = spacy.load('es')

# End Imports

# Definitions
def load_data(
    data_directory, 
    image_location, 
    scraped_posts_location,
    gold_labels_location,
    ocr_data_location, 
    outout_location, 
    use_embeddings
):
    def valid_text(x):
        if pd.isnull(x):
            return False
        elif isinstance(x, (int, long, float)):
            return False
        else:
            return True

    def get_spacy_doc(x, bolCheckHash):
        #if x is not None:
            #print x
        if valid_text(x):
            if bolCheckHash:
                return nlp(x.replace("#", "HASH"))
            else: 
                return nlp(x)
        else:
            return nlp(u"")

    def merge_text_sources(row):
        if row["Scraped Dict"] is None:
            return row["Content"]
        elif pd.notnull(row["Content"]):
            return row["Content"]
        else:
            return row["Scraped Dict"].get("post_text")

    def parse_ocr(x):
        if isinstance(x, dict):
            if "Text" in x:
                return x["Text"].decode('utf-8', "ignore")
            else:
                return x[" Text"].decode('utf-8', "ignore")
        elif x is not None:
            return x.decode('utf-8', "ignore")
        else:
            return None

    def get_face_from_image(vertex):
        image_path = image_location+vertex+".jpg"
        if Path(image_path).is_file():
            image = face_recognition.load_image_file(image_path)
            return len(face_recognition.face_locations(image)) > 0
        else:
            return False    

    def remove_bad_rows(df, num_checks = 1):

        df = df.set_index("Vertex", drop=False)
        df["Bad Row"] = df.apply(lambda row: False if row["Vertex Type"] == "Post"\
                                         or pd.notnull(row["Parent"]) else True, axis=1)
        for i in range(num_checks):
            bad_rows_df = df[df["Bad Row"] == True]
            df["Bad Row"] = df.apply(lambda row: True if row["Bad Row"] is True\
                                               or row["Parent"] in bad_rows_df.index else False, axis=1)

        return df.query("Bad Row == False")        

    def build_frame_from_raw_files(file):
        # open file
        spreadsheet = pd.ExcelFile(data_directory+file)

        # Gather edge connections
        edges = spreadsheet.parse("Edges", header=1, usecols=[0,1])
        #Create dictionary
        edges_dict = {}
        for index, row in edges.iterrows():
            edges_dict[row["Vertex 1"]] = row["Vertex 2"]

        # Gather vertices
        vertices = spreadsheet.parse("Vertices", header=1)

        #Add the appropriate dictionary entry to each row by iterating through each row
        # and adding the row plus the new Parent column to a new dataset
        joined_df = vertices
        joined_df['Parent'] = joined_df['Vertex'].map(edges_dict)
        joined_df["Group"] = file[:file.find("_")-1]
        joined_df["Month"] = file[-8:-5]  
        return joined_df

    #use this to flatten the comment matrix so all sub-child, sub-sub-child, etc comments 
    #are related directly to the top post
    def get_ultimate_parent(vertex_id):
        if vertex_id in post_df.index:
            return vertex_id
        else:
            get_ultimate_parent(comment_df.loc[vertex_id, "Parent"])


    #this is the function you can use to get a list of common rows in a dataframe "apply" function
    def get_comments(post_id):
        if post_id in comment_grp_df.groups:
            #print [post["Text"].decode('utf-8') for _,post in comment_df.get_group(post_id).to_dict(orient='index').items()]
            return comment_grp_df.get_group(post_id).to_dict(orient='index')
        else:
            return {}        

    # for initial keyword-based crime type classification model
    crime_keyword_dict = {
        'secuestro':'Kidnapping', 
        'raptar':'Kidnapping', 
        'extorsion': 'Extortion',
        'homicida': 'Homicide',
        'matar': 'Homicide',
        'estafa': 'Extortion',
        'fraude' : 'Extortion',
        'desaparecer' : 'Disappearance',
        'desaparecido' : 'Disappearance',
        'localizon' : 'Disappearance',
        'buscar' : 'Disappearance',
        'busca' : 'Disappearance',
        'violar' : 'Rape',
        'droga' : 'Drugs',
        'narco' : 'Drugs',                
        'robo': 'Robbery', 
        'robar': 'Robbery',
        'robada': 'Robbery',
        'robándo' : 'Robbery',
        'quitar' : 'Robbery',
        'arrebato': 'Robbery',
        'ratero': 'Robbery',
        'rata' : 'Robbery',
        'asalto' : 'Robbery',
        'asaltante' : 'Robbery',
        'asalto' : 'Robbery',
        'asaltar' : 'Robbery',
        'pistola' : 'Violence',
        'arma' : 'Violence', 
        'delincuente' : 'Violence', 
        'violencia' : 'Violence', 
        'golpear' : 'Violence',
        'golpiza' : 'Violence'
    }    

    def get_crime_type(token_lst):
        for key, value in keyword_dict.items():
            if key in token_lst:
                return value
        return "Noncriminal"

    gold_label_dict = {
        "Robbery":0, 
        "Disappearance":1, 
        "Homicide":2,
        "Violence":3, 
        "Extortion":4, 
        "Noncriminal":5
    }

    gold_id_dict = {
        0:"Robbery",
        1:"Disappearance",
        2: "Homicide",
        3:"Violence", 
        4:"Extortion", 
        5:"Noncriminal"
    }

    def build_disappearance_labels(x):
        if x is None or pd.isnull(x):
            return None
        elif x == "Disappearance":
            return 1
        else:
            return 0

    # End Definitions

    # Data Loading

    # Parse each file in the selected directory
    files = [x for x in os.listdir(data_directory) if x.endswith('.xlsx')]
    df_list = [build_frame_from_raw_files(file) for file in files]
    raw_df = pd.concat(df_list)

    # get files from scraped posts
    # use "Caption_Scraping_Script.py" to build this directory
    with open(scraped_posts_location) as f:
        scraped_posts = json.load(f)
    df["Scraped Dict"] = df["Vertex"].apply(lambda x: scraped_posts.get(x))

    # import OCR data
    with open(ocr_data_location) as f:
        ocr_data = pd.read_csv(f, index_col="Id").to_dict(orient="index")

    #get gold standard labels
    with open(gold_labels_location) as f:
        gold_crime_labels = pd.read_csv(f, index_col="Id").to_dict(orient="index")   

    # Data Processing

    df = remove_bad_rows(raw_df)

    #generate fields from merged dict
    df["FB Image Description"] = df["Scraped Dict"].apply(lambda x:
                                                              None if x is None
                                                              else x["image_description"]
                                                         )
    df["Text of Shared Link"] = df["Scraped Dict"].apply(lambda x:
                                                             None if x is None
                                                             else x["link_header"]
                                                        )
    df["Merged Text"] = df.apply(lambda row: merge_text_sources(row), axis=1)

    df["Text of Shared Link"] = df.apply(lambda x: x["Text of Shared Link"]
                                                 if x["Vertex"] not in ocr_data
                                                 else parse_ocr(ocr_data[x["Vertex"]]),
                                                 axis=1)

    df["gold_label"] = df["Vertex"].apply(lambda x: gold_crime_labels.get(x)["Crime Type"]
                                                      if x in gold_crime_labels else None)

    df["URL"]  = df.apply(lambda row: row["Post URL"] if not pd.isnull(row["Post URL"])\
                                  else row["Comment URL"],\
                                  axis=1)
    df["Date"] = df.apply(lambda row: row["Post Date"] if not pd.isnull(row["Post Date"])\
                                  else row["Comment Date"],\
                                  axis=1)
    df["Image URL"]  = df.apply(lambda row: row["Image"] if not pd.isnull(row["Image"])\
                                  else row["Attachment URL"],\
                                  axis=1)
    df["Has Text"] = df["Merged Text"].apply(lambda x: valid_text(x))
    df["Has Image"] = df["Image URL"].apply(lambda x: pd.notnull(x))
    df["Has Shared Article"] = df["Text of Shared Link"].apply(lambda x: pd.notnull(x))
    df["Total Shares"] = df["Total Shares"].apply(lambda x: int(x) if not pd.isnull(x) else 0)
    df["Popularity Measure"] = df.apply(lambda row:\
                                                row["Total Shares"] + row["Total Comments"] + row["Total Likes"],\
                                                axis=1)
    # may take awhile
    df["Person in Image"] = df["Vertex"].apply(lambda x: get_face_from_image(x))

    # may take awhile
    df["Doc"] = df["Merged Text"].apply(lambda x: get_spacy_doc(x, True))
    df["Link Description Doc"] = df["Text of Shared Link"].apply(lambda x: get_spacy_doc(x, False))
    df["Image Description Doc"] = df["FB Image Description"].apply(lambda x: get_spacy_doc(x, False))

    df = df.rename(columns={"Merged Text":"Text", "Vertex Type":"Type", "Vertex":"Post Id"})
    df = df.loc[:,["Parent", "Group", "Month", "Type", "Date", "Post Id", "gold_label",\
                           "Has Text","Has Image","Has Shared Article",\
                           "Doc", "Text","Link Description Doc","Image Description Doc",\
                           "URL","Image URL",\
                           "Popularity Measure","Total Likes", "Total Shares", "Total Comments"]]

    # Just comments dataframe
    comment_df = df.query("Type == 'Comment'")
    comment_df["Post Id"] = comment_df["Parent"].apply(lambda x: get_ultimate_parent(x))
    comment_grp_df = comment_df.groupby("Post Id")

    # Just posts dataframe
    post_df = df.query("Type == 'Post'")
    post_df = post_df.drop("Parent", axis=1)

    post_df["Gold_Crime_Type_Label"] = post_df["Post Id"].apply(lambda x: gold_crime_labels.get(x)["Crime Type"]\
                                                      if x in gold_crime_labels else None)

    post_df["Gold_Disappearance_Label"] = post_df["Gold_Crime_Type_Label"].apply(lambda x: build_disappearance_labels(x))

    post_df["Keyword_Predicted_Crime_Type"]= post_df.apply(\
                           lambda row:"Shared Link" if not row["Has Text"]\
                                         else get_crime_type([token.lemma_.lower() for token in row["Doc"]]),\
                                         axis=1)

    post_df["Label_Id"] = post_df["Gold_Crime_Type_Label"].apply(lambda x: gold_label_dict.get(x, None))
    post_df = post_df.sort_values(by="Label_ID",axis=0, na_position="last")
    
    # reincorporate comments
    post_df["Comments"] = post_df["Post Id"].apply(lambda x: get_comments(x))
    post_df["Comment_Docs"] = post_df["Comments"].apply(lambda x: [value["Doc"] for value in x.itervalues()])
    post_df["Comment_Docs"].head(10)


    # Start building vectorized features for ML

    cleanedComments = list(post_df["comment_docs"].apply(lambda lst: " ".join(\
                                                            [" ".join(\
                                                                      [token.lemma_ for token in doc\
                                                                       if not token.is_punct and not token.is_stop\
                                                                       and (token.pos_ in\
                                                                            ['NOUN', 'ADJ', 'VERB', 'ADV']\
                                                                            or token.ent_type != ''\
                                                                           )\
                                                                      ]\
                                                                     )\
                                                             for doc in lst]\
                                                           )))

    cleanedAllPosts = list(post_df["Doc"].apply(lambda x: " ".join([token.lemma_ for token in x\
                                                                       if not token.is_punct and not token.is_stop\
                                                                       and (token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']\
                                                                            or token.ent_type != '')])))

    # Adapted from http://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
    goldVectorizer = CountVectorizer(analyzer='word',       
                                 min_df=0.01,
                                 lowercase=True,
                                 token_pattern='[a-zA-Z]{3,}',
                                 ngram_range=(1,2)
                                )
    all_data_vectorized = goldVectorizer.fit_transform(cleanedAllPosts)

    # Materialize the sparse data
    all_data_dense = all_data_vectorized.todense()
    word_count_df = pd.DataFrame(data=all_data_dense, columns=["NGRAM_"+x for x in goldVectorizer.get_feature_names()])

    commentVectorizer = TfidfVectorizer(analyzer='word',       
                                 min_df=0.01,
                                 lowercase=True,
                                 token_pattern='[a-zA-Z]{3,}',
                                 ngram_range=(1,2)
                                )
    comments_vectorized = commentVectorizer.fit_transform(cleanedComments)

    # Materialize the sparse data
    comment_data_dense = comments_vectorized.todense()
    comment_word_count_df = pd.DataFrame(data=comment_data_dense, columns=["COMMENT_"+x for x in\
                                                                           commentVectorizer.get_feature_names()])

    #word features for link descriptions
    cleanedLinkDescriptions = list(
        post_df["Link Description Doc"].apply(
        lambda x:
        " ".join([token.lemma_ for token in x
                  if not token.is_punct
                  and not token.is_stop
                  and (
                      token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']
                       or token.ent_type != ''
                  )
                 ]
                )
        )
    )
    linkDescriptionVectorizer = CountVectorizer(analyzer='word',       
                                 min_df=3,
                                 #max_df=0.1,
                                 lowercase=True,
                                 token_pattern='[a-zA-Z0-9]{3,}',
                                 ngram_range=(1,2)
                                )

    link_data_vectorized = linkDescriptionVectorizer.fit_transform(cleanedLinkDescriptions)

    # Materialize the sparse data
    link_data_dense = link_data_vectorized.todense()
    link_word_count_df = pd.DataFrame(data=link_data_dense,\
                                            columns=["LINK_"+x for x in linkDescriptionVectorizer.get_feature_names()])

    #word features for image descriptions
    cleanedImageDescriptions = list(post_df["Image Description Doc"].apply(lambda x:\
                                                                                  " ".join([token.lemma_ for token in x\
                                                                                            if not token.is_punct\
                                                                                            and not token.is_stop])))
    imageDescriptionVectorizer = CountVectorizer(analyzer='word',       
                                 min_df=3,
                                 lowercase=True,
                                 token_pattern='[a-zA-Z0-9]{1,}',
                                 ngram_range=(1,2),
                                 stop_words = ["people", "person", "baby", "on", "available",
                                                   "and", "outdoor", "one"]
                                )

    image_data_vectorized = imageDescriptionVectorizer.fit_transform(cleanedImageDescriptions)

    # Materialize the sparse data
    image_data_dense = image_data_vectorized.todense()
    image_word_count_df = pd.DataFrame(data=image_data_dense,\
                                            columns=["IMAGE_"+x for x in imageDescriptionVectorizer.get_feature_names()])

    # Optional - use word embeddings as features
    if use_embeddings:
        post_vectors = []
        for text in post_df["Doc"]:
            if text.vector.shape[0]==0:
                post_vectors.append(np.zeros(shape=(1,384)))
            else:
                post_vectors.append(np.max(np.array([token.vector/(token.vector_norm or 1.) for token in text\
                                                     if not token.is_punct and not token.is_stop]), axis=0)\
                                    +\
                                    np.min(np.array([token.vector/(token.vector_norm or 1.) for token in text\
                                                     if not token.is_punct and not token.is_stop]), axis=0)
                                   )

    comment_vectors = []
    for comment_list in post_df["Comment Docs"]:
        temp = []
        if len(comment_list)==0:
            comment_vectors.append(np.zeros(shape=(384)))
        else:
            for text in comment_list:
                if text.vector.shape[0]==0:
                    temp.append(np.zeros(shape=(384)))
                else:
                    temp.append(np.nanmax(np.array([token.vector/(token.vector_norm or 1.) for token in text\
                                                         if not token.is_punct and not token.is_stop]), axis=0)\
                                        +\
                                        np.nanmin(np.array([token.vector/(token.vector_norm or 1.) for token in text\
                                                         if not token.is_punct and not token.is_stop]), axis=0))
            comment_vectors.append(np.nanmax(np.vstack(np.array(temp)), axis=0)\
                                   + np.nanmin(np.vstack(np.array(temp)), axis=0)
                                  )



    existing_features_df = post_df.reset_index()
    existing_features_df["Has Text"] = existing_features_df["Has Text"].apply(lambda x: 1 if x else 0)
    existing_features_df["Has Image"] = existing_features_df["Has Image"].apply(lambda x: 1 if x else 0)
    existing_features_df["Has Shared Article"] = existing_features_df["Has Shared Article"].apply(lambda x: 1 if x else 0)
    existing_features_df["num_words"] = existing_features_df["Doc"].apply(lambda x: len(x))
    existing_features_df["avg_word_length"] = existing_features_df["Doc"].apply(lambda x: 0 if len(x) == 0 else np.mean([len(token.text) for token in x]))
    existing_features_df["has_url"] = existing_features_df["Text"].apply(lambda x:\
                                                              1 if x is not None and any([val in x for val in \
                                                                       ["http",".com",".co.","www","://"]]) else 0)
    existing_features_df["Total Shares"] = existing_features_df["Total Shares"].apply(lambda x: x if pd.notnull(x) else 0)
    existing_features_df["Total Likes"] = existing_features_df["Total Likes"].apply(lambda x: x if pd.notnull(x) else 0)
    existing_features_df["Total Comments"] = existing_features_df["Total Comments"].apply(lambda x: x if pd.notnull(x) else 0)

    existing_features_df["Outdoors in Image"] = existing_features_df["Image Description"]\
    .apply(lambda x: 1 if pd.notnull(x) and any(kw in x for kw in ["outdoor","tree","sky","water", "mountain", "nature"]) else 0)

    existing_features_df["Multiple People in Image"] = existing_features_df["Image Description"]\
    .apply(lambda x: 1 if pd.notnull(x) and any(kw in x for kw in ["2","3","4","5","6", "two", "three", "four"]) else 0)

    existing_features_df["Animal in Image"] = existing_features_df["Image Description"]\
    .apply(lambda x: 1 if pd.notnull(x) and any(kw in x for kw in ["cat","dog","horse"]) else 0)

    existing_features_df["Text in Image"] = existing_features_df["Image Description"]\
    .apply(lambda x: 1 if pd.notnull(x) and any(kw in x for kw in ["text", "meme"]) else 0)

    return_df = pd.concat([existing_features_df, word_count_df, link_word_count_df, image_word_count_df,\
                         comment_word_count_df],axis=1)

    if use_embeddings:
        post_vectors = pd.DataFrame(post_vectors, columns=["EMBEDDING_"+str(i) for i in range(384)])
        comment_vectors = pd.DataFrame(comment_vectors, columns=["COMMENT_EMBEDDING_"+str(i) for i in range(384)])
        return_df = pd.concat([return_df, post_vectors, comment_vectors],axis=1)

    return_df = return_df.drop(["Comment Docs", "Image Description Doc", "Image Description", 
                                "Image URL", "Doc", "Link Description Doc"], axis = 1)

    return_df.to_csv(output_location+"/cleaned_data.csv")

if __name__ == "__main__":
    
    data_directory = sys.argv[1]
    image_location = sys.argv[2]
    scraped_posts_location = sys.argv[3]
    gold_labels_location = sys.argv[4]
    ocr_data_location = sys.argv[5]
    outout_location = sys.argv[6]
    use_embeddings = sys.argv[7]
    
    load_data(data_directory, image_location, scraped_posts_location, gold_labels_location,
             ocr_data_location, use_embeddings)
    