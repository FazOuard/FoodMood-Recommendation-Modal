import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from .database import ApiSQLEngine



# Chargement des données
data = pd.read_sql_query("SELECT * FROM users_preferences", ApiSQLEngine)
plat= pd.read_sql_query("SELECT * FROM plat", ApiSQLEngine)
historique = pd.read_sql_query("SELECT * FROM historique", ApiSQLEngine)

# Fonction pour nettoyer les chaînes de caractères
def clean_data(x):
    return str(x).lower().replace(" ", "")

# Colonnes textuelles et numériques
text_features = ['Plat_prefere', 'Plat_consome', 'type_cuisine', 'regime_alimentaire',
                 'Allergies', 'Allergie_specification', 'sport_pratique',
                 'Aimer_Plat_marocain', 'type_viande_prefere', 'regime_raison',
                 'dejeuner_preference', 'Nationalite', 'Region', 'Vegeterien_question',
                 'Sport_question', 'Poids_etat', 'duree_preparation']
numeric_features = ['age', 'Poids', 'Taille']

for feature in text_features:
    data[feature] = data[feature].apply(clean_data)

# Normalisation des colonnes numériques
scaler = MinMaxScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

def get_user_history(user_id, historique=historique):
    user_history = historique[historique['user_id'] == user_id]
    return ' '.join(user_history['plat'].tolist())

def create_user_soup(x):
    user_id = x.name
    user_history = get_user_history(user_id)  
    weighted_history = ' '.join([plat for plat in user_history.split() for _ in range(3)])  
    text_data = ' '.join([str(x[feature]) for feature in text_features])
    numeric_data = ' '.join([str(round(x[feature], 2)) for feature in numeric_features])
    return text_data + ' ' + numeric_data + ' ' + weighted_history

data['soup'] = data.apply(create_user_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

#User based recommendation
# Fonction pour obtenir des recommandations basées sur un utilisateur
def recommend_for_user(user_index, cosine_sim=cosine_sim, data=data):
    if user_index >= len(data):
       user_index=user_index-1
        
    sim_scores = list(enumerate(cosine_sim[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  
    similar_users = [i[0] for i in sim_scores]
    
    recommended_plates = []
    for u in similar_users:
        plates = data.iloc[u]['Plat_prefere'].split(',') + data.iloc[u]['Plat_consome'].split(',')
        recommended_plates.extend(plates)

    user_plates = data.iloc[user_index]['Plat_prefere'].split(',') + data.iloc[user_index]['Plat_consome'].split(',')
    recommended_plates = list(set(recommended_plates) - set(user_plates))

    return recommended_plates[:10]  


#plats recommandation

def plat_recommended(user_id):
    recommendations = recommend_for_user(user_id)
    plat_resultat = []

   
    def clean_search_title(title):
        return str(title).lower().replace(" ", "")
    
    for i in recommendations:
      
        plat_info = plat[plat["Titre"].apply(clean_search_title) == clean_search_title(i)]

        if not plat_info.empty:
            for _, row in plat_info.iterrows():
                
                if pd.isna(row["Image"]) or pd.isna(row["Recette"]):
                    print(f"Invalid entry skipped: {row['Titre']}")
                    continue

                
                plat_id = row["id"]
                if pd.isna(plat_id):
                    print(f"ID manquant pour le plat : {row['Titre']}")
                    continue

                plat_resultat.append({
                    "Titre": row["Titre"],
                    "Image": row["Image"]
                })
        else:
            print(f"Aucune recette trouvée pour {i}")

    return plat_resultat


#Content based recommendation
# Fonction pour obtenir des recommandations basées sur le contenu
def content_based_recommendations(train_data, item_id, top_n=10):
    train_data['Ingredients'] = train_data['Ingredients'].fillna('')
    if item_id not in train_data['id'].values:
        print(f"Item with id '{item_id}' not found in the training data.")
        return []

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Ingredients'])

    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    item_index = train_data[train_data['id'] == item_id].index[0]

    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    top_similar_items = similar_items[1:top_n+1]

    recommended_item_indices = [x[0] for x in top_similar_items]

    recommended_items_details = train_data.iloc[recommended_item_indices][['Titre', 'Image']]

    recommendations = recommended_items_details.to_dict(orient='records')

    return recommendations

# Hybrid recommendations

def hybrid_recommendations(train_data, target_user_id, item_id, top_n=10):
    content_based_rec = content_based_recommendations(train_data, item_id, top_n)
    content_based_rec = pd.DataFrame(content_based_rec)

    collaborative_filtering_rec = plat_recommended(target_user_id)
    collaborative_filtering_rec = pd.DataFrame(collaborative_filtering_rec)

    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()

    return hybrid_rec