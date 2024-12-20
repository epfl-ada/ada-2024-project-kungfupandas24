import pandas as pd
import os
import json
import pickle

class CMUDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_plot_summaries(self):
        path = os.path.join(self.data_dir, "CMU", "plot_summaries.txt")
        return pd.read_csv(path, delimiter="\t", names=["Wikipedia_movie_ID", "Plot Summaries"])

    def load_movie_metadata(self):
        path = os.path.join(self.data_dir, "CMU", "movie.metadata.tsv")
        return pd.read_csv(path, delimiter='\t', names=[
            "Wikipedia_movie_ID", "Freebase_movie_ID", "Movie_name", "Movie_release_date",
            "Movie_box_office_revenue", "Movie_runtime", "Movie_languages", "Movie_countries", "Movie_genres"])

    def load_character_metadata(self):
        path = os.path.join(self.data_dir, "CMU", "character.metadata.tsv")
        df = pd.read_csv(path, delimiter="\t", header=None)
        df.columns = [
            "Wikipedia_movie_ID", "Freebase_movie_ID", "Movie_release_date", 
            "Character_name", "Actor_date_of_birth", "Actor_gender", "Actor_height", 
            "Actor_ethnicity", "Actor_name", "Actor_age_at_movie_release",
            "Freebase_character_actor_map_ID", "Freebase_character_ID", "Freebase_actor_ID"]
        
        df.drop(columns=['Actor_height', 'Actor_ethnicity', 'Character_name'], inplace=True)

        return df
    
    def load_character_types(self):
        path = os.path.join(self.data_dir, "CMU", "tvtropes.clusters.txt")
        df = pd.read_csv(path, delimiter='\t', names= ['Character_type', 'Metadata'])
        metadata = df['Metadata'].apply(json.loads).apply(pd.Series)
        df = pd.concat([df.drop(columns=['Metadata']), metadata], axis=1)
        df = df.rename(columns={'char':'Character_name', 'id': 'Freebase_character_actor_map_ID', 'movie':'Movie_name', 'actor':'Actor_name'})

        return df
        
class IMDBDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_ratings(self):
        path = os.path.join(self.data_dir, "IMDB", "title.ratings.tsv")
        return pd.read_csv(path, delimiter="\t", na_values="\\N", names=["tconst", "Average_ratings", "Num_votes"], low_memory=False)

    def load_basics(self):
        path = os.path.join(self.data_dir, "IMDB", "title.basics.tsv")
        df = pd.read_csv(path, delimiter="\t", na_values="\\N", names=[
            "tconst", "Title_type", "Primary_title", "Original_title", "Is_adult",
            "Start_year", "End_year", "Movie_runtime", "Movie_genres"], low_memory=False)
        
        # FIlter movies only 
        df = df[df["Title_type"] == "movie"]

        return df


    def load_crew(self):
        path = os.path.join(self.data_dir, "IMDB", "title.crew.tsv")
        return pd.read_csv(path, delimiter="\t", na_values="\\N", names=["tconst", "Directors", "Writers"], low_memory=False)

class KaggleDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_kaggle(self):
        path = os.path.join(self.data_dir, "KaggleSingh","movie_dataset.csv")
        Kaggle_df = pd.read_csv(path, na_values="\\N", low_memory=False)

        Kaggle_df = Kaggle_df[["budget", "genres", "original_title", "popularity", "production_companies", "production_countries", "revenue"]]
        
        # Clean and convert financials to float for consistency with the movie_df
        Kaggle_df["budget"] = Kaggle_df["budget"].astype(float)
        Kaggle_df["revenue"] = Kaggle_df["revenue"].astype(float)

        # Drop incorrect data rows where the budget is less than 1000
        Kaggle_df = Kaggle_df[Kaggle_df["budget"] >= 1000]

        return Kaggle_df

class NumbersDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    def load_numbers(self):
        path = os.path.join(self.data_dir, "TheNumbers","tn.movie_budgets.csv")
        return pd.read_csv(path, na_values="\\N", low_memory=False)

class BechdelDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    def load_bechdel(self):
        path = os.path.join(self.data_dir, "Bechdel test","bechdel_test_movies.csv")
        df = pd.read_csv(path, names=["Id", "Year", "Title", "Bechdel_rating", "tconst"], header=None)
        df = df.drop(index=0)
        
        #To make the format appropriate for merge with the final_df
        df["tconst"] = df["tconst"].apply(lambda x : "tt"+str(x))
        df["Bechdel_rating"] = pd.to_numeric(df["Bechdel_rating"])
        
        return df

class DialogueDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_movie_dialogue(self):
        path = os.path.join(self.data_dir, "Dialogue","meta_data7.csv")
        df = pd.read_csv(path, names=["Script_id", "tconst", "Title", "Year", "Gross", "Lines_Data"], header=0, encoding="ISO-8859-1")
        df.drop(columns=["Gross", "Lines_Data"], inplace=True)
        return df
        
    def load_character_dialogue(self):
        path = os.path.join(self.data_dir, "Dialogue","character_list5.csv")
        df = pd.read_csv(path, names=["Script_id", "Character_name", "Words", "Gender", "Age"], header=0, encoding="ISO-8859-1", na_values= "?")
        df.dropna(subset=["Character_name", "Age"], inplace=True)
        return df


class ZeroShotResultsLoader:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def load_results(self, label):
        path = os.path.join(self.results_dir, "NLP", f"zeroshot_labels{label}.csv")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"An error occurred while loading the results: {e}")

class ClusteringResultsLoader:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def load_results(self, labels):
        path = os.path.join(self.results_dir, "NLP", f"clustering_results.csv")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"An error occurred while loading the results: {e}")

class ClusteringResultsLoader:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def load_csv(self):
        path = os.path.join(self.results_dir, "NLP", "clustering_results.csv")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}")
            return None

    def load_dictionary(self):
        path = os.path.join(self.results_dir, "NLP", "clustering_details.pkl")
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the dictionary: {e}")
            return None

    def load_clusters(self):
        df = self.load_csv()
        if df is None:
            return None, None

        cluster_details = self.load_dictionary()
        if cluster_details is None:
            return df, None

        return df, cluster_details
