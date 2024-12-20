import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EDA:
    def __init__(self, dataframe, numeric_columns=[
        "Average_ratings",
        "Num_votes",
        "Movie_release_date",
        "Movie_budget",
        "Final_movie_revenue",
        "ROI",
        "Movie_runtime",
        "Female_actors",
        "Male_actors",
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

    
    def plot_histograms(self, variables, title, bins=15, layout=(1, None)):
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
            sns.histplot(self.dataframe, x=col, kde=True, stat="count", ax=axes[i], bins=bins)
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

        plt.suptitle("Figure 5: Boxplot of Variables")
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

        genres_series = genres_series.apply(
            lambda x: "action/adventure" if x in ["action", "adventure"] else x
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
        threshold = 50
        if popular_genres is None:
            raise ValueError("`popular_genres` is required.")

        # Ensure popular_genres are lowercase
        popular_genres = set(map(str.lower, popular_genres))

        # Apply function to extract the main genre
        self.dataframe["Main_genre"] = (
            self.dataframe[column_name]
            .apply(lambda x: self.get_first_genre(x, popular_genres))
            .str.replace(" ", "_")
        )

        popular_genres_counts = self.dataframe["Main_genre"].value_counts()

        # Filter out genres with counts < threshold
        filtered_popular_counts = popular_genres_counts[popular_genres_counts >= threshold]

        filtered_popular = filtered_popular_counts.index.tolist()

        self.dataframe["Main_genre"] = (
            self.dataframe[column_name]
            .apply(lambda x: self.get_first_genre(x, filtered_popular))
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

    def frequency_actors_gender(self):
        bins = np.histogram_bin_edges(
        np.concatenate((self.dataframe["Female_actors"], self.dataframe["Male_actors"])),
        bins=70
        )
        female_hist, _ = np.histogram(self.dataframe["Female_actors"], bins=bins)
        male_hist, _ = np.histogram(self.dataframe["Male_actors"], bins=bins)

        # Use bin centers for alignment
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bar_width = (bins[1] - bins[0]) / 3  # Adjust width to fit side by side


        fig = go.Figure()

        # Add Female actors bar chart
        fig.add_trace(go.Bar(
            x=bin_centers - bar_width,  # Shift to the left
            y=female_hist,
            name="Female Actors",
            marker=dict(color='magenta'),
            width=bar_width  # Set bar width
        ))

        # Add Male actors bar chart
        fig.add_trace(go.Bar(
            x=bin_centers + bar_width,  # Shift to the right
            y=male_hist,
            name="Male Actors",
            marker=dict(color='blue'),
            width=bar_width 
        ))

        fig.update_layout(
            title="Histogram of Female and Male Actors",
            xaxis_title="Log of Number of Actors",
            yaxis_title="Frequency",
            barmode="group",  
            template="plotly_white",
            legend_title="Actor Gender"
        )

        fig.show()
 
    
    def plot_female_percentage(self, columns=["Movie_release_date"], plot_type="Line"):
        """
        Plot the average female actor percentage per column of choice using Plotly.

        Args:
            column (str): The column name to group by (default is "Movie_release_date").
            plot_type (str): The type of plot ("line", "bar", or "Interactif by genre"). Default is "line".

        Returns:
            None
        """
        # Group data by release date and calculate mean female actor percentage
        female_percentage_df = (
            self.dataframe
            .groupby(columns)["Female_actor_percentage"]
            .mean()
            .reset_index()
        )

        if len(columns) == 1:
            columns = columns[0]

            if plot_type == "Line":
                # Create a line plot
                fig = px.line(
                    female_percentage_df,
                    x=columns,
                    y="Female_actor_percentage",
                    title="Average Female Actor Percentage Per Year",
                    labels={columns: columns.replace("_", " "), 'Female_actor_percentage': 'Average Female Actor Percentage (%)'}
                )

                # Update the trace color and layout
                fig.update_traces(line_color="orange")
                fig.update_layout(
                    xaxis=dict(range=[1980, female_percentage_df["Movie_release_date"].max()]),
                    template="plotly_white"
                )

                fig.show()

            elif plot_type == "Bar":
                # Plot using Plotly Express
                fig = px.bar(
                    female_percentage_df,
                    x=columns,
                    y='Female_actor_percentage',
                    title='Average Female Actor Percentage by Movie Genre',
                    labels={columns: columns.replace("_", " "), 'Female_actor_percentage': 'Average Female Actor Percentage (%)'}
                )

                # Update bar color
                fig.update_traces(marker_color="orange")
                fig.update_layout(template="plotly_white")

                fig.show()

            else:
                # Handle incorrect plot type
                raise ValueError("Invalid plot_type. If columns length is 1, expected 'Line' or 'Bar'.")
        else:
            if plot_type == "Interactif by genre":
                # Group data by release date and genre
                female_percentage_per_year_genre_df = (
                    self.dataframe
                    .groupby(columns)["Female_actor_percentage"]
                    .mean()
                    .reset_index()  # This will retain 'Movie_main_genre' in the DataFrame
                )

                # Get the unique genres
                unique_genres = female_percentage_per_year_genre_df["Movie_main_genre"].unique()

                fig = go.Figure()

                # Add a trace for each genre
                for genre in unique_genres:
                    genre_data = female_percentage_per_year_genre_df[female_percentage_per_year_genre_df["Movie_main_genre"] == genre]
                    fig.add_trace(
                        go.Bar(
                            x=genre_data["Movie_release_date"],
                            y=genre_data["Female_actor_percentage"],
                            name=genre,
                            marker_color="orange",  # Set bar color to orange
                            visible=False  # Initially make all traces invisible
                        )
                    )

                # Make the first genre visible by default
                fig.data[0].visible = True

                # Add dropdown menu for selecting genres
                dropdown_buttons = [
                    dict(
                        label=genre,
                        method="update",
                        args=[
                            {"visible": [i == idx for i in range(len(unique_genres))]},  # Update visibility
                            {"title": f"Average Female Actor Percentage Per Year ({genre})"},
                        ],
                    )
                    for idx, genre in enumerate(unique_genres)
                ]

                # Update layout with dropdown menu
                fig.update_layout(
                    updatemenus=[
                        dict(
                            active=0,  # Default genre
                            buttons=dropdown_buttons,
                            direction="down",
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.15,
                            yanchor="top",
                        )
                    ],
                    title="Average Female Actor Percentage Per Year by Genre",
                    xaxis_title="Movie Release Date",
                    yaxis_title="Average Female Percentage (%)",
                    bargap=0,
                    template="plotly_white"
                )

                fig.show()
            else:
                # Handle incorrect plot type
                raise ValueError("Invalid plot_type. If columns length greater than 1, expected 'Interactif by genre'.")



    def plot_gender_comparison(self, columns=["log_ROI", "Movie_success", "Normalized_Rating"], interactive=False):
        """
        Plot the number of Female and Male actors vs the columns of choice as subplots.

        Args:
            columns (list): List of column names for movie success metrics.
            interactive (bool): Whether plot is interactive and allows changing between genres.

        Returns:
            None
        """
        if not interactive:
            fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True,
                                subplot_titles=[f"Average {col.replace('_', ' ')} vs. Number of Actors" for col in columns])

            # Create subplots
            for i, column in enumerate(columns):
                female_data = self.dataframe.groupby("Female_actors")[column].mean().reset_index()
                male_data = self.dataframe.groupby("Male_actors")[column].mean().reset_index()

                # Add Female data
                fig.add_trace(go.Bar(
                    x=female_data["Female_actors"],
                    y=female_data[column],
                    name="Female Actors",
                    marker=dict(color='rgb(255, 165, 0)'),
                    showlegend=(i == 0)  # Show legend only once
                ), row=i+1, col=1)

                fig.add_trace(go.Scatter(
                    x=female_data["Female_actors"],
                    y=female_data[column],
                    mode='lines',
                    name=f"Female Peaks",
                    line=dict(color='rgb(255, 140, 0)', width=2, dash='dash'),  # Pink dashed line
                    marker=dict(size=6, symbol='circle'),
                    showlegend=False
                ), row=i+1, col=1)

                # Add Male data
                fig.add_trace(go.Bar(
                    x=male_data["Male_actors"],
                    y=male_data[column],
                    marker=dict(color='rgb(139, 69, 19)'),
                    name="Male Actors",
                    showlegend=(i == 0)  
                ), row=i+1, col=1)

                fig.add_trace(go.Scatter(
                    x=male_data["Male_actors"],
                    y=male_data[column],
                    mode='lines',
                    name=f"Male Peaks",
                    line=dict(color='rgb(160, 82, 45)', width=2, dash='dot'),  # Blue dotted line
                    marker=dict(size=6, symbol='circle'),
                    showlegend=False
                ), row=i+1, col=1)

            fig.update_layout(
                height=300 * len(columns),
                title_text="Gender Comparison Across Metrics",
                xaxis_title="Number of Actors",
                template="plotly_white"
            )


            # Add Y-axis labels
            for i, column in enumerate(columns):
                fig.update_yaxes(title_text=column.replace('_', ' '), row=i+1, col=1)

            # Add X-axis labels and graduations for interactive plot
            for i in range(len(columns)):
                fig.update_xaxes(title_text="Average Number of Actors", row=i+1, col=1, showgrid=True, ticks="outside")


            fig.show()
        else:
            genres = [genre for genre in self.dataframe['Movie_main_genre'].unique() if not pd.isna(genre)]

            # Create subplots
            fig = make_subplots(
                rows=len(columns),
                cols=1,
                shared_xaxes=True,
                subplot_titles=[f"Average {col.replace('_', ' ')} vs. Number of Actors" for col in columns]
            )

            traces_per_genre = len(columns) * 4  # Each genre has two traces per column (Female and Male)

            buttons = []  # To hold the genre selection buttons

            for genre_index, genre in enumerate(genres):
                # Create visibility list for all traces
                visibility = [False] * (traces_per_genre * len(genres))

                for i, column in enumerate(columns):
                    # Filter data by genre
                    genre_data = self.dataframe[self.dataframe['Movie_main_genre'] == genre]

                    female_data = genre_data.groupby("Female_actors")[column].mean().reset_index()
                    male_data = genre_data.groupby("Male_actors")[column].mean().reset_index()

                    # Add Female data as a bar plot
                    fig.add_trace(go.Bar(
                        x=female_data["Female_actors"],
                        y=female_data[column],
                        name=f"Female Actors ({genre})",
                        marker=dict(color='rgb(255, 165, 0)'),  # Light pastel pink
                        visible=(genre_index == 0),  # Initially visible for the first genre
                        showlegend=(i == 0)  # Show legend only for the first column
                    ), row=i+1, col=1)

                    # Add a line following the peaks of Female data
                    fig.add_trace(go.Scatter(
                        x=female_data["Female_actors"],
                        y=female_data[column],
                        mode='lines',
                        name=f"Female Peaks ({genre})",
                        line=dict(color='rgb(255, 140, 0)', width=2, dash='dash'),  # Pink dashed line
                        marker=dict(size=6, symbol='circle'),
                        visible=(genre_index == 0),
                        showlegend=False
                    ), row=i+1, col=1)

                    # Add Male data as a bar plot
                    fig.add_trace(go.Bar(
                        x=male_data["Male_actors"],
                        y=male_data[column],
                        name=f"Male Actors ({genre})",
                        marker=dict(color='rgb(139, 69, 19)'),  # Light pastel blue
                        visible=(genre_index == 0),  # Initially visible for the first genre
                        showlegend=(i == 0)  # Show legend only for the first column
                    ), row=i+1, col=1)

                    # Add a line following the peaks of Male data
                    fig.add_trace(go.Scatter(
                        x=male_data["Male_actors"],
                        y=male_data[column],
                        mode='lines',
                        name=f"Male Peaks ({genre})",
                        line=dict(color='rgb(160, 82, 45)', width=2, dash='dot'),  # Blue dotted line
                        marker=dict(size=6, symbol='circle'),
                        visible=(genre_index == 0),
                        showlegend=False
                    ), row=i+1, col=1)



                # Update visibility list for the current genre
                visibility[genre_index * traces_per_genre:(genre_index + 1) * traces_per_genre] = [True] * traces_per_genre

                # Add button for this genre
                buttons.append(dict(
                    label=genre,
                    method="update",
                    args=[{"visible": visibility},  # Update visibility of traces
                        {"title": f"Gender Comparison for Genre: {genre}"}]
                ))

            fig.update_layout(
                updatemenus=[dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    x=0.9,
                    xanchor="left",
                    y=1.1,
                    showactive=True
                )],
                height=300 * len(columns),
                title_text=f"Gender Comparison Across Metrics",
                xaxis_title="Number of Actors",
                template="plotly_white"
            )

            # Add Y-axis labels
            for i, column in enumerate(columns):
                fig.update_yaxes(title_text=column.replace('_', ' '), row=i+1, col=1)

            # Add X-axis labels and graduations for interactive plot
            for i in range(len(columns)):
                fig.update_xaxes(title_text="Average Number of Actors", row=i+1, col=1, showgrid=True, ticks="outside")

            fig.show()