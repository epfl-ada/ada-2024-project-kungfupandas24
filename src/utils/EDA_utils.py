import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import numpy as np
import ast

class EDA:
    def __init__(self, dataframe, numeric_columns=[
        "Average_ratings",
        "Num_votes",
        "Movie_release_date",
        "Final_movie_revenue",
        "ROI",
        "Movie_runtime",
        "Female_actors",
        "Male_actors",
        "Movie_success"
    ]):
        """
        Initialize the EDA (Exploratory Data Analysis) class.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to analyze.
            numeric_columns (list): List of numeric columns to analyze.
                Defaults to a predefined set of numeric columns.
        """
        self.dataframe = dataframe
        self.numeric_columns = numeric_columns

    def summary(self):
        """
        Generate summary statistics for the numeric columns.

        Returns:
            pd.DataFrame: A DataFrame containing min, max, mean, std, and median for numeric columns.
        """
        # Check if the numeric columns exist in the DataFrame
        missing_columns = [col for col in self.numeric_columns if col not in self.dataframe.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")

        # Generate summary statistics
        summary_table = self.dataframe[self.numeric_columns].agg(["min", "max", "mean", "std", "median"]).T
        summary_table.columns = ["Min", "Max", "Mean", "SD", "Median"]
        summary_table = summary_table.round(2)

        return summary_table
    
    def summary_bis(self):
        """
        Generate summary statistics for the numeric columns available in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing min, max, mean, std, and median for numeric columns.
        """
        # Filter numeric columns that exist in the DataFrame
        existing_columns = [col for col in self.numeric_columns if col in self.dataframe.columns]
        missing_columns = [col for col in self.numeric_columns if col not in self.dataframe.columns]

        # Log ignored columns
        if missing_columns:
            print(f"Warning: The following columns are missing and will be ignored: {missing_columns}")

        # Generate summary statistics for existing columns
        if not existing_columns:
            raise ValueError("No numeric columns are available in the DataFrame for summarization.")
        
        summary_table = self.dataframe[existing_columns].agg(["min", "max", "mean", "std", "median"]).T
        summary_table.columns = ["Min", "Max", "Mean", "SD", "Median"]
        summary_table = summary_table.round(2)

        return summary_table


    def plot_histograms(self, dep_var=["Average_ratings", "Final_movie_revenue", "ROI", "Movie_success"], bins=15):
        """
        Plot histograms for the specified dependent variables.

        Args:
            dep_var (list): List of column names to plot histograms for.
            bins (int): Number of bins for the histograms. Default is 15.

        Returns:
            None
        """
        fig, axes = plt.subplots(nrows=1, ncols=len(dep_var), figsize=(18, 6))
        fig.suptitle("Figure 1: Histogram of Dependent Variables", fontsize=14)
        axes = axes.flatten()

        # Plot histograms for each dependent variable
        for i, col in enumerate(dep_var):
            if col not in self.dataframe.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")
            if i < len(axes):  # Ensure we do not exceed the number of axes
                ax = axes[i]
                sns.histplot(self.dataframe, x=col, kde=True, stat="density", ax=ax, bins=bins)
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

        # Hide any extra subplots if there are more axes than columns
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Make space for the title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_log_transformed_histograms(self, columns_to_transform=["Final_movie_revenue", "ROI"], bins=15):
        """
        Plot histograms for log-transformed dependent variables.

        Args:
            columns_to_transform (list): List of column names to apply log transformation and plot histograms for.
            bins (int): Number of bins for the histograms. Default is 15.

        Returns:
            None
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram for each log-transformed variable
        sns.histplot(self.dataframe["log_Final_movie_revenue"], kde=True, edgecolor="black", color="skyblue", bins=bins, ax=axes[0])
        axes[0].set_title("Histogram of log-transformed Final_movie_revenue")
        axes[0].set_xlabel("log_Final_Movie_revenue")
        axes[0].set_ylabel("Frequency")

        sns.histplot(self.dataframe["log_ROI"], kde=True, edgecolor="black", color="skyblue", bins=bins, ax=axes[1])
        axes[1].set_title("Histogram of log-transformed ROI")
        axes[1].set_xlabel("log_ROI")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_independent_histograms(self, indep_var=["Num_votes", "Movie_release_date", "Movie_runtime", "Female_actors", "Male_actors"], bins=15):
        """
        Plot histograms for the specified independent variables.

        Args:
            indep_var (list): List of column names to plot histograms for.
            bins (int): Number of bins for the histograms. Default is 15.

        Returns:
            None
        """
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
        fig.suptitle("Figure 2: Histogram of Independent Variables", fontsize=14)

        axes = axes.flatten()

        # Plot histograms for each independent variable
        for i, col in enumerate(indep_var):
            if col not in self.dataframe.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")
            sns.histplot(self.dataframe, x=col, kde=True, stat="density", ax=axes[i], bins=bins)
            axes[i].set_title(f"Histogram of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        # Hide any extra subplots if there are more axes than columns
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_log_transformed_independent_histograms(self, columns_to_transform=["Num_votes", "Movie_runtime", "Female_actors", "Male_actors"], bins=15):
        """
        Apply log transformation to skewed independent variables and plot histograms.

        Args:
            columns_to_transform (list): List of column names to apply log transformation and plot histograms for.
            bins (int): Number of bins for the histograms. Default is 15.

        Returns:
            None
        """

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6))
        fig.suptitle("Figure 3: Histograms of Log-Transformed Independent Variables", fontsize=14)

        sns.histplot(self.dataframe["log_Num_votes"], kde=True, ax=axes[0], color="skyblue", bins=bins)
        axes[0].set_title("Log of Num_votes")

        sns.histplot(self.dataframe["log_Movie_runtime"], kde=True, ax=axes[1], color="skyblue", bins=bins)
        axes[1].set_title("Log of Movie_runtime")

        sns.histplot(self.dataframe["log_Female_actors"], kde=True, ax=axes[2], color="skyblue", bins=bins)
        axes[2].set_title("Log of Female_actors")

        sns.histplot(self.dataframe["log_Male_actors"], kde=True, ax=axes[3], color="skyblue", bins=bins)
        axes[3].set_title("Log of Male_actors")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def plot_histograms_combined(self, variables, title, bins=15, layout=(1, None)):
        """
        Plot histograms for the specified variables (dependent or independent).

        Args:
            variables (list): List of column names to plot histograms for.
            title (str): Title of the figure.
            bins (int): Number of bins for the histograms. Default is 15.
            layout (tuple): Tuple indicating the number of rows and columns for subplots.
                            If columns are None, it is calculated automatically.

        Returns:
            None
        """
        # Calculate the number of columns if not provided
        if layout[1] is None:
            n_cols = len(variables)
        else:
            n_cols = layout[1]

        n_rows = layout[0]

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 4))
        fig.suptitle(title, fontsize=14)

        # Flatten axes for easier indexing
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        # Plot histograms for each variable
        for i, col in enumerate(variables):
            if col not in self.dataframe.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")
            sns.histplot(self.dataframe, x=col, kde=True, stat="density", ax=axes[i], bins=bins)
            axes[i].set_title(f"Histogram of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        # Hide any extra subplots if there are more axes than variables
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def plot_boxplots(self, numeric_columns=None):
        """
        Plot boxplots for the specified numeric columns.

        Args:
            numeric_columns (list): List of column names to plot boxplots for. If None, use all numeric columns.

        Returns:
            None
        """
        if numeric_columns is None:
            numeric_columns = self.numeric_columns

        plt.figure(figsize=(24, 8))
        for i, col in enumerate(numeric_columns, 1):
            if col not in self.dataframe.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")
            plt.subplot(3, 3, i)
            sns.boxplot(data=self.dataframe, x=col)
            plt.title(f"Boxplot of {col}")

        plt.suptitle("Figure 4: Boxplot of Variables")
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self):
        """
        Calculate and visualize the correlation matrix of numeric variables.

        Returns:
            None
        """
        correlation_matrix = self.dataframe[self.numeric_columns].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Figure 6: Correlation Matrix of Numeric Variables")
        plt.show()
        
    def filter_and_count(self, column_name="Movie_genres", threshold=1000):
        """
        Filter and count genres from the specified column, removing genres below the threshold.

        Args:
            column_name (str): The column containing genres.
            threshold (int): Minimum count to include a genre.

        Returns:
            pd.Series: Filtered genre counts.
        """
        genres_series = (
            self.dataframe[column_name]
            .str.replace(r"[\[\]\']", "", regex=True)  # Clean the string further for this application
            .str.split(", ")  # Split the genre strings into lists for easier manipulation, using ", " as the delimiter
            .explode()  # Expand the lists into separate rows with one genre per row
            .str.strip()  # Strip leading and trailing whitespace from each genre
            .str.lower()
        )

        # Get the count of each genre
        genre_counts = genres_series.value_counts()

        # Filter out genres with counts < threshold
        filtered_genre_counts = genre_counts[genre_counts >= threshold]

        # Plot the distribution of genres as a vertical bar chart
        plt.figure(figsize=(8, 4))
        sns.barplot(y=filtered_genre_counts.values, x=filtered_genre_counts.index)
        plt.title("Figure 7: Distribution of Genres")
        plt.ylabel("Count")
        plt.xlabel("Genre")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        
        print(f"There are {len(filtered_genre_counts)} genres with more that {threshold} occurences.")

        return filtered_genre_counts
    
    def analyze_popular_genres(self, column_name="Movie_genres", filtered_genre_counts=None):
        """
        Analyze rows containing popular genres and their proportion in the dataset.

        Args:
            column_name (str): The column containing genres.
            filtered_genre_counts (pd.Series): Filtered genre counts.

        Returns:
            None
        """
        if filtered_genre_counts is None:
            raise ValueError("`filtered_genre_counts` is required. Use `filter_and_count_genres` to generate it.")

        # List of genres that meet the threshold
        popular_genres = filtered_genre_counts.index.tolist()

        # Remove all special characters from the strings to allow proper regex matching
        no_special_char = map(re.escape, popular_genres)

        # Create regex pattern using the "OR" operator
        pattern_genre = "|".join(no_special_char)

        # Check thanks to the regex pattern if any of the popular genres are in each column of the dataframe
        rows_with_popular_genres = self.dataframe[column_name].str.contains(pattern_genre, case=False, na=False)

        print(f"There are {rows_with_popular_genres.sum()} movies that belong to at least one of the {len(filtered_genre_counts)} most popular genres.",
              f"\nThere were {len(self.dataframe)} movies in the dataset before this operation,", 
              f"meaning we lost {(len(self.dataframe) - rows_with_popular_genres.sum())/len(self.dataframe):.2%} of the movies.")
        
        # Keeping only the rows with popular genres
        self.dataframe = self.dataframe[rows_with_popular_genres]
        
        return popular_genres
    
    def get_first_genre(self,genres, popular_genres):
        for genre in map(str.strip, genres.split(",")):  # Split genres by "," and remove spaces
            if genre.lower() in popular_genres:  # Check against popular genres
                return genre  # Return the first match
        return np.nan
        
    def get_main_genres(self, column_name="Movie_genres", popular_genres=None):
        """
        Get the main genre for each movie and plot the distribution of main genres.

        Args:
            column_name (str): The column containing genres.
            popular_genres (set): A set of popular genres to prioritize. Must be provided.

        Returns:
            None
        """
        if popular_genres is None:
            raise ValueError("`popular_genres` is required.")

        # Ensure popular_genres are lowercase
        popular_genres = set(map(str.lower, popular_genres))
        
        # Replacing "action" if present, with the more general "action/adventure"
        popular_genres = ["action/adventure" if x == "action" else x for x in popular_genres]

        # Apply function to extract the main genre
        self.dataframe["Main_genre"] = (
            self.dataframe[column_name]
            .str.replace("action", "action/adventure") # Grouping action and adventure movies into action/adventure
            .str.replace("adventure", "action/adventure")
            .apply(lambda x: self.get_first_genre(x, popular_genres))
            .str.replace(" ", "_")
        )

        # Plot the distribution of main genres
        genre_counts = self.dataframe["Main_genre"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        plt.bar(genre_counts.index, genre_counts.values, edgecolor="black")
        plt.title("Distribution of Main Genres", fontsize=16)
        plt.xlabel("Main Genres", fontsize=14)
        plt.ylabel("Number of Movies", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    def analyze_languages(self, column_name="Movie_languages"):
        """
        Analyze and visualize the top 5 languages by occurrences in the dataset.

        Args:
            column_name (str): The column containing languages.

        Returns:
            pd.Series: Counts of all languages.
        """
        language_series = (
            self.dataframe[column_name]
            .str.replace(r"[\[\]']", "", regex=True)
            .str.split(", ")
            .explode()
            .str.strip()
            .str.lower()
            .str.replace(" language", "")
        )

        language_count = language_series.value_counts()

        # Plot the top 5 languages by occurrences
        language_count.head(5).plot(kind="bar", figsize=(10, 6))
        plt.xlabel("Languages")
        plt.ylabel("Occurrences")
        plt.title("Top 5 Languages by Occurrences")
        plt.xticks(rotation=0)
        plt.yscale("log")
        plt.tight_layout()
        plt.show()
    
    def categorize_languages(self, column_name="Movie_languages"):
        """
        Categorize movies as "English_only" or "Not_only_english" based on their language and create a dummy variable.

        Args:
            column_name (str): The column containing languages.

        Returns:
            int: The number of movies available in other languages.
        """
        # Categorize rows as "English_only" or "Not_only_english"


        #self.dataframe["Language_Category"] = self.dataframe[column_name].apply(
       #    lambda x: "English_only" if x.lower() == "english language" else "Not_only_english"
        #)
        self.dataframe["Language_Category"] = self.dataframe[column_name].fillna("").apply(
            lambda x: "English_only" if x.lower() in ["english", "english language"] else "Not_only_english"
        )

        # Create dummy variables and drop the reference category
        self.dataframe = pd.get_dummies(self.dataframe, columns=["Language_Category"], drop_first=True)

        # Rename the column for clarity
        self.dataframe.rename(columns={"Language_Category_Not_only_english": "Is_not_only_english"}, inplace=True)

        # Count the number of movies available in other languages
        not_only_eng_count = self.dataframe["Is_not_only_english"].sum()

        print(
            f"There are {not_only_eng_count} movies that are available in other languages and "
            f"{len(self.dataframe) - not_only_eng_count} movies that are only available in English."
        )
        
    def analyze_countries(self, column_name="Movie_countries"):
        """
        Analyze and visualize the top 10 countries by occurrences in the dataset.

        Args:
            column_name (str): The column containing country information.

        Returns:
            pd.Series: Counts of all countries.
        """
        # Drop rows where the column is empty
        self.dataframe = self.dataframe.drop(self.dataframe[self.dataframe[column_name] == ""].index)

        # Process the country information
        country_series = (
            self.dataframe[column_name]
            .str.replace(r"[\[\]']", "", regex=True)
            .str.split(", ")
            .explode()
            .str.strip()
            .str.lower()
        )

        # Count the occurrences of each country
        country_count = country_series.value_counts()

        # Plot the top 10 countries by occurrences
        country_count.head(10).plot(kind="bar", figsize=(10, 6))
        plt.xlabel("Countries")
        plt.ylabel("Occurrences")
        plt.title("Top 10 Countries by Occurrences")
        plt.xticks(rotation=65)
        plt.yscale("log")
        plt.tight_layout()
        plt.show()

    def categorize_countries(self, column_name="Movie_countries"):
        """
        Categorize movies as "USA_movies" or "Non_USA_movies" based on their country and create a dummy variable.

        Args:
            column_name (str): The column containing country information.

        Returns:
            None
        """
        # Categorize rows as "USA_movies" or "Non_USA_movies"
        self.dataframe["Country_Category"] = self.dataframe[column_name].fillna("").apply(
            lambda x: "USA_movies" if x.lower() in["united states of america","United States"] else "Non_USA_movies"
        )


        # Create dummy variables and drop the reference category
        self.dataframe = pd.get_dummies(self.dataframe, columns=["Country_Category"], drop_first=True)

        # Rename the column for clarity
        self.dataframe.rename(columns={"Country_Category_USA_movies": "Is_USA_movie"}, inplace=True)

        # Count the number of USA movies and movies from other countries
        count_usa_movies = self.dataframe["Is_USA_movie"].sum()
        count_other_countries_movie = len(self.dataframe) - count_usa_movies

        print(
            f"There are {count_usa_movies} USA movies and {count_other_countries_movie} movies from other countries."
        )

    def categorize_countries_bis(self, column_name="Movie_countries"):
        """
        Categorize movies as "USA_movies" or "Non_USA_movies" based on their country and create a dummy variable.

        Args:
            column_name (str): The column containing country information.

        Returns:
            None
        """
        # Categorize rows as "USA_movies" or "Non_USA_movies"
        self.dataframe["Country_Category"] = self.dataframe[column_name].fillna("").apply(
            lambda x: "USA_movies" if "united states" in str(x).lower() else "Non_USA_movies"
        )

        # Create dummy variables and drop the reference category
        self.dataframe = pd.get_dummies(self.dataframe, columns=["Country_Category"], drop_first=True)

        # Check if the expected column exists before renaming
        if "Country_Category_USA_movies" in self.dataframe.columns:
            self.dataframe.rename(columns={"Country_Category_USA_movies": "Is_USA_movie"}, inplace=True)
        else:
            print("Warning: 'Country_Category_USA_movies' column not found. Check the input data.")

        # Count the number of USA movies and movies from other countries
        if "Is_USA_movie" in self.dataframe.columns:
            count_usa_movies = self.dataframe["Is_USA_movie"].sum()
            count_other_countries_movie = len(self.dataframe) - count_usa_movies

            print(
                f"There are {count_usa_movies} USA movies and {count_other_countries_movie} movies from other countries."
            )
        else:
            print("Error: 'Is_USA_movie' column was not created.")

        
    def extract_production_names(self, companies_str):
        """
        Helper function to extract production company names from a string.

        Args:
            companies_str (str): A string representation of a list of dictionaries.

        Returns:
            list: A list of production company names.
        """
        if pd.isna(companies_str):  
            return []
        try:
            companies_list = ast.literal_eval(companies_str)
            return [company["name"] for company in companies_list]
        except (ValueError, SyntaxError, KeyError):
            return []
        
    def analyze_production_companies(self, column_name="Production_companies", threshold=100):
        """
        Extract, clean, and analyze production company data, filtering by a specified threshold.

        Args:
            column_name (str): The column containing production company information.
            threshold (int): The minimum number of occurrences for a production company to be included.

        Returns:
            pd.Series: Filtered counts of production companies meeting the threshold.
        """

        # Apply extraction function to clean production company data
        self.dataframe["Production_companies_cleaned"] = self.dataframe[column_name].apply(self.extract_production_names)

        # Filter out rows with no production companies
        self.dataframe = self.dataframe[self.dataframe["Production_companies_cleaned"].str.len() > 0]

        # Explode the cleaned list into individual rows for analysis
        production_series = self.dataframe["Production_companies_cleaned"].explode()

        # Count occurrences of each production company
        production_count = production_series.value_counts()

        print(f"There are {len(production_count)} unique production companies.")

        # Plot the production company counts (Top 10 for clarity)
        production_count.head(10).plot(
            kind="bar", 
            figsize=(14, 6), 
            edgecolor="black"
        )
        plt.title("Top 10 Production Companies by Occurrences", fontsize=16)
        plt.xlabel("Production Companies", fontsize=14)
        plt.ylabel("Occurrences", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Filter production companies by threshold
        filtered_production_count = production_count[production_count >= threshold]
        
        return filtered_production_count

    def merge_production_data(self, filtered_production_count, prod_companies, company_translation_reversed=None):
        """
        Merge filtered production company data with additional company details.

        Args:
            filtered_production_count (pd.DataFrame): DataFrame containing production company counts.
            prod_companies (pd.DataFrame): DataFrame containing additional company details.
            company_translation_reversed (dict, optional): Dictionary mapping original names to their standardized versions.

        Returns:
            pd.DataFrame: Merged DataFrame containing production company data and additional details.
        """
        # Reset index and rename columns for filtered production data
        filtered_production_count = filtered_production_count.reset_index()
        filtered_production_count.columns = ["Production_companies_cleaned", "Occurrences"]
        
        # Apply name translation to the production companies if a translation dictionary is provided
        if company_translation_reversed:
            prod_companies["Company Name"] = prod_companies["Company Name"].replace(company_translation_reversed)

        # Merge the production company data with additional details
        merged_data = filtered_production_count.merge(
            prod_companies, 
            left_on="Production_companies_cleaned", 
            right_on="Company Name", 
            how="left"
        )

        return merged_data
    
    def calculate_box_office(self,companies_list, box_office):

        box_office["Total Worldwide Box Office"] = box_office["Total Worldwide Box Office"].replace(
        {"\\$": "", ",": ""}, regex=True).astype(float)
        if not companies_list:
            return 0  
        return box_office.loc[box_office["Company Name"].isin(companies_list), "Total Worldwide Box Office"].sum()
    
    def unify_columbia_revenue(self, box_office_df, source_column="Company Name", 
                           target_column="Production_companies_cleaned", box_office_column="Total Worldwide Box Office"):
        """
        Ensure the Box Office for 'Columbia Pictures Corporation' matches 'Columbia Pictures'.

        Args:
            box_office_df (pd.DataFrame): DataFrame containing box office datas.
            source_column (str): Column where there is 'Columbia Pictures'.
            target_column (str): Column where there is 'Columbia Pictures Corporation'.
            box_office_column (str): Colonne contenant les revenus du box office.

        Returns:
            None
        """
        box_office_df.loc[
            box_office_df[target_column] == "Columbia Pictures Corporation", box_office_column
        ] = box_office_df.loc[
            box_office_df[source_column] == "Columbia Pictures", box_office_column
        ].values[0]
     

    def filter_movies_with_box_office(self, box_office_column="Box_office_companies"):
        """
        Filter out rows where the box office revenue is null and display the remaining count.

        Args:
            box_office_column (str): The column containing box office revenue.

        Returns:
            None
        """
        # Nombre de films avant filtrage
        initial_count = len(self.dataframe)
        
        # Filtrer les films avec une valeur non nulle dans box_office_column
        self.dataframe = self.dataframe[self.dataframe[box_office_column].notnull()]
        remaining_count = len(self.dataframe)

        # Afficher un message
        print(f"After removing rows without Box Office revenue, {remaining_count} movies remain out of {initial_count}.")

