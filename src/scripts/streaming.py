import pandas as pd
import numpy as np
import unicodedata

# Fonction pour normaliser le texte (noms ou titres)
def normalize_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        text = "".join(
            char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
        )
        text = " ".join(text.split())  
    return text

def get_streaming_dataframe():
    # We load streaming platform data files
    netflix = pd.read_csv("data/data_streaming/netflix_titles.csv", na_values="\\N")
    disney = pd.read_csv("data/data_streaming/disney_plus_titles.csv", na_values="\\N")
    amazon = pd.read_csv("data/data_streaming/amazon_prime_titles.csv", na_values="\\N")
    hulu = pd.read_csv("data/data_streaming/hulu_titles.csv", na_values="\\N")
    
    #Then, we keep only movies
    netflix_movies = netflix[netflix["type"] == "Movie"].copy()
    amazon_movies = amazon[amazon["type"] == "Movie"].copy()
    hulu_movies = hulu[hulu["type"] == "Movie"].copy()
    disney_movies = disney[disney["type"] == "Movie"].copy()
    netflix_movies["Platform"] = "Netflix"
    hulu_movies["Platform"] = "Hulu"
    amazon_movies["Platform"] = "Amazon"
    disney_movies["Platform"] = "Disney"

    #We combine the dataframes
    all_movies = pd.concat([netflix_movies, hulu_movies, amazon_movies, disney_movies], ignore_index=True)
    all_movies_cleaned = all_movies.dropna(subset=["cast"])

    #The goal in this part is to find the gender of the streaming characters
    #We load metadata and gender files to find the gender of the streaming actors thanks to their first_name
   
    character_metadata = pd.read_csv("data/CMU/character.metadata.tsv", delimiter="\t", header=None)
    character_metadata.columns = [
        "Wikipedia_movie_ID", "Freebase_movie_ID", "Movie_release_date", "Character_name",
        "Actor_date_of_birth", "Actor_gender", "Actor_height", "Actor_ethnicity",
        "Actor_name", "Actor_age_at_movie_release", "Freebase_character_actor_map_ID",
        "Freebase_character_ID", "Freebase_actor_ID"]
    
    character_metadata[["first_name", "last_name"]] = character_metadata["Actor_name"].astype(str).str.split(" ", n=1, expand=True)
    actor_cmu_df = character_metadata[["first_name", "last_name", "Actor_gender"]].copy()
    actor_cmu_df.columns = ["first_name", "last_name", "gender"]

    
    with open("data/data_streaming/male.txt", "r") as male_file:
        male_names = [line.strip() for line in male_file.readlines()]

    with open("data/data_streaming/female.txt", "r") as female_file:
        female_names = [line.strip() for line in female_file.readlines()]

    male_df = pd.DataFrame({"first_name": male_names, "last_name": "", "gender": "M"})
    female_df = pd.DataFrame({"first_name": female_names, "last_name": "", "gender": "F"})

    #We combine all names and gender data into a single DataFrame
    names_df = pd.concat([actor_cmu_df, male_df, female_df], ignore_index=True)
    names_df["first_name_normalized"] = names_df["first_name"].apply(normalize_text)

    all_movies_cleaned["cast"] = all_movies_cleaned["cast"].str.split(", ")
    streaming_actors = all_movies_cleaned.explode("cast").dropna(subset=["cast"]).copy()
    streaming_actors["cast"] = streaming_actors["cast"].astype(str)
    streaming_actors[["first_name", "last_name"]] = streaming_actors["cast"].str.split(" ", n=1, expand=True)
    streaming_actors["first_name_normalized"] = streaming_actors["first_name"].apply(normalize_text)

    
    gender_dict = dict(zip(names_df["first_name_normalized"], names_df["gender"]))

    streaming_actors["gender"] = streaming_actors["first_name_normalized"].map(gender_dict).fillna("unknown")
    unknown_gender_df = streaming_actors[streaming_actors["gender"] == "unknown"]
    streaming_actors = streaming_actors[streaming_actors["gender"] != "unknown"]

    #We add additional gender data from gender_found_df to improve the gender identification
    #This data comes from the website genderize.io, which classifies gender from names
    #We take unknown_gender_df, then we uploaded on genderize.io to have 
   
    gender_found_df = pd.read_csv("data/data_streaming/gender_actors_found.csv", sep=";")
    gender_found_df = gender_found_df[["first_name", "last_name", "Gender"]]
    gender_found_df.rename(columns={"Gender": "gender"}, inplace=True)
    gender_found_df["gender"] = gender_found_df["gender"].replace({"male": "M", "female": "F", "unknown": np.nan})

    streaming_actors_final = pd.concat([streaming_actors, gender_found_df], ignore_index=True)
    actor_gender_dict = {f"{row['first_name']} {row['last_name']}": row["gender"] for _, row in streaming_actors_final.iterrows()}

    def count_genders(cast, actor_gender_dict):
        male_count = 0
        female_count = 0
        not_found_count = 0
        for actor in cast.split(", "):
            if actor in actor_gender_dict:
                if actor_gender_dict[actor] == "M":
                    male_count += 1
                elif actor_gender_dict[actor] == "F":
                    female_count += 1
            else:
                not_found_count += 1
        return pd.Series([male_count, female_count, not_found_count])


    all_movies_cleaned["cast"] = all_movies_cleaned["cast"].apply(lambda x: ", ".join(x))
    all_movies_cleaned[["male_count", "female_count", "not_found_count"]] = all_movies_cleaned["cast"].apply(
        lambda cast: count_genders(cast, actor_gender_dict))
    
    streaming = all_movies_cleaned[all_movies_cleaned['not_found_count'] == 0]

    ratings_df = pd.read_csv("data/IMDB/title.ratings.tsv", delimiter="\t", na_values="\\N")
    basics_df = pd.read_csv("data/IMDB/title.basics.tsv", delimiter="\t", na_values="\\N", low_memory=False)
    merged_df = pd.merge(
        ratings_df[["tconst", "averageRating", "numVotes"]],
        basics_df[["tconst", "primaryTitle", "startYear"]],
        on="tconst",
        how="inner")

    
    all_movies_cleaned["title_normalized"] = all_movies_cleaned["title"].apply(normalize_text)
    merged_df["primaryTitle_normalized"] = merged_df["primaryTitle"].apply(normalize_text)

    streaming = pd.merge(
        all_movies_cleaned,
        merged_df[["primaryTitle_normalized", "startYear", "averageRating", "numVotes"]],
        left_on=["title_normalized", "release_year"],
        right_on=["primaryTitle_normalized", "startYear"],
        how="inner")


    
    streaming.drop(columns=["title_normalized", "primaryTitle_normalized", "startYear", "not_found_count"], inplace=True)

    def is_adult_rating(rating):
        if rating in ["TV-MA", "R", "18+", "NC-17", "AGES_18_"]:
            return 1
        else:
            return 0

    streaming['duration'] = pd.to_numeric(streaming['duration'].str.replace(' min', '', regex=False), errors='coerce')
    streaming["Is_Adult"] = streaming["rating"].apply(is_adult_rating)
    streaming = streaming[streaming['duration'] != 0]
    streaming.dropna(subset=['duration'], inplace=True)

    new_column_names = ["Show_id", "Type","Movie_name", "Director", "Cast", "Movie_countries", "Date_added", "Movie_release_date", "Age_ratings","Movie_runtime","Movie_genres", "Description", "Platform","Male_actors", "Female_actors", "Average_Rating", "Num_Votes", "Is_Adult"]
    streaming.columns = new_column_names


    return streaming
