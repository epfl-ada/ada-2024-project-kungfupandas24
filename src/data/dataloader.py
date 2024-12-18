import pandas as pd
import os

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