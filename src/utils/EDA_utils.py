import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import os


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
        bins=40
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
            marker=dict(color='rgb(255, 165, 0)'),
            width=bar_width  # Set bar width
        ))

        # Add Male actors bar chart
        fig.add_trace(go.Bar(
            x=bin_centers + bar_width,  # Shift to the right
            y=male_hist,
            name="Male Actors",
            marker=dict(color='rgb(139, 69, 19)'),
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
        #fig.write_html("Frequency_Gender.html")
 
    
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
                
                # Add a polynomial fit of degree 2
                x = female_percentage_df[columns]
                y = female_percentage_df["Female_actor_percentage"]
                polynomial = np.poly1d(np.polyfit(x, y, 2))
                y_fit = polynomial(x)

                # Add polynomial fit to the plot
                fig.add_scatter(
                    x=x,
                    y=y_fit,
                    mode="lines",
                    name="Polynomial trend",
                    line=dict(color="blue", dash="dash")
                )
                
                # Update the trace color and layout
                fig.update_traces(line_color="orange")
                fig.update_layout(
                    xaxis=dict(range=[1980, female_percentage_df["Movie_release_date"].max()]),
                    template="plotly_white",
                    showlegend=False
                )
                fig.write_html("HTML_TIME_SERIES.html")
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
                #fig.write_html("Per_Year.html")
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
                #fig.write_html("By_Genre_Per_Year.html")
            else:
                # Handle incorrect plot type
                raise ValueError("Invalid plot_type. If columns length greater than 1, expected 'Interactif by genre'.")



    def plot_gender_comparison(self, columns=["log_ROI", "Normalized_Rating"], interactive=False):
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
            #fig.write_html("Gender_across_Metrics.html")
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
            #fig.write_html("By_genre_Gender_across_Metrics.html")


    def analyze_female_actors_by_genre(self):
        """
        Analyze and visualize the average number and percentage of female actors by genre.
        """
        # Filter the data
        df_filtered = self.dataframe[['Movie_release_date', 'Male_actors', 'Female_actors', 'Movie_genres']].copy()
        df_filtered['Movie_release_date'] = pd.to_numeric(df_filtered['Movie_release_date'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['Movie_release_date', 'Male_actors', 'Female_actors', 'Movie_genres'])
        df_filtered['Total_actors'] = df_filtered['Male_actors'] + df_filtered['Female_actors']
        df_filtered['Percentage_female'] = (df_filtered['Female_actors'] / df_filtered['Total_actors']) * 100

        # Explode genres
        df_exploded = df_filtered.assign(Movie_genres=df_filtered['Movie_genres'].str.split(',')).explode('Movie_genres')
        df_exploded['Movie_genres'] = df_exploded['Movie_genres'].str.strip()

        # Remove Uncategorized genre
        df_exploded = df_exploded[df_exploded['Movie_genres'].str.lower() != 'uncategorized']

        # Group by genre
        df_grouped_number = df_exploded.groupby('Movie_genres', as_index=False)['Female_actors'].mean()
        df_grouped_percentage = df_exploded.groupby('Movie_genres', as_index=False)['Percentage_female'].mean()

        # Create the plot
        fig = go.Figure()

        # Average number of female actors by genres
        fig.add_trace(go.Bar(
            x=df_grouped_number['Movie_genres'],
            y=df_grouped_number['Female_actors'],
            name='Average Number of Female Actors',
            marker=dict(color='royalblue')
        ))

        #  Average percentage of female actors by genres
        fig.add_trace(go.Bar(
            x=df_grouped_percentage['Movie_genres'],
            y=df_grouped_percentage['Percentage_female'],
            name='Average Percentage of Female Actors',
            marker=dict(color='lightcoral'),
            visible=False
        ))

        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="Average Number of Female Actors",
                             method="update",
                             args=[{"visible": [True, False]},
                                   {"yaxis": {"title": "Average Number of Female Actors"}}]),
                        dict(label="Average Percentage of Female Actors",
                             method="update",
                             args=[{"visible": [False, True]},
                                   {"yaxis": {"title": "Average Percentage of Female Actors (%)"}}])
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.05,  
                    yanchor="top"
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title={
                "text": "Average Number or Percentage of Female Actors by Genre",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Movie Genre", yaxis_title="Value", template="plotly_white"
        )

        fig.show()
        #fig.write_html("average_number_or_percentage_of_female.html")


    def analyze_countries_plotly(self, column_name="Movie_countries"):
        """
        Analyze and visualize the top 10 represented countries in the dataset.

        Args:
            column_name (str): The column containing country information.
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

        # Count the occurrences for each country
        country_count = country_series.value_counts()

        # Top 10 countries
        top_10_countries = country_count.head(10)

        # Bar chart
        fig = px.bar(top_10_countries,
            x=top_10_countries.index,
            y=top_10_countries.values,
            labels={"x": "Countries", "y": "Occurrences" },
            title="Top 10 Countries by Occurrences",
        )

        fig.update_layout(
            xaxis_title="Countries",
            yaxis_title="Occurrences",
            xaxis_tickangle=65,
            yaxis_type="log",
            template="plotly_white",
        )

        
        fig.show()
        #fig.write_html("analyze_countries_plotly.html")

        #return country_count

    def calculate_average_rating_by_country(self, country_column="Movie_countries", rating_column="Average_ratings"):
        """
        Calculates the average rating for each country.
        Args:
            country_column (str): Column containing country names.
            rating_column (str): Column containing ratings.
        Returns:
            pd.DataFrame: DataFrame with countries and their average ratings.
        """
        if country_column not in self.dataframe.columns or rating_column not in self.dataframe.columns:
            raise ValueError("The specified columns do not exist in the DataFrame.")
        
        # Separate countries and explode rows, then reset the index to handle duplicates
        exploded_df = self.dataframe.copy()
        exploded_df = exploded_df.assign(**{country_column: exploded_df[country_column].str.split(',') }).explode(country_column).reset_index(drop=True)
        
        # Strip whitespace from country names
        exploded_df[country_column] = exploded_df[country_column].str.strip()

        # Calculate average rating by country
        avg_ratings = exploded_df.groupby(country_column)[rating_column].mean().reset_index()
        avg_ratings.columns = ["Country", "Average_Rating"]

        return avg_ratings
    
    def plot_average_rating_by_country(self, country_column="Movie_countries", rating_column="Average_ratings", color_scale="Blues"):
    
        """
        Crée et affiche une carte choroplèthe pour visualiser les données par pays.
        Args:
            country_column (str): Column with countries
            value_column (str): Name of the column containing the values to visualize.
            title (str): Plot title.
            color_scale (str): Color scale for the visualization.
        """

        avg_ratings = self.calculate_average_rating_by_country(country_column, rating_column)

        fig = px.choropleth(
            avg_ratings,
            locations="Country",
            locationmode="country names",
            color="Average_Rating",
            color_continuous_scale=color_scale,
            title="Average Rating by Country")
        
        fig.show()
        #fig.write_html("plot_average_rating_by_country.html")

    def plot_female_percentage_evolution(self):
        """
        Generates a bar plot with the evolution of the percentage of female actors over the years> 1970
        grouped by the movie genre(without Uncategorized) genres.
        """
        # Calculate the percentage of female actors
        self.dataframe["Female_actor_percentage"] = (self.dataframe["Female_actors"] / (self.dataframe["Female_actors"] 
                                                                                        + self.dataframe["Male_actors"]))*100

        # Replace NaN percentages (from division by zero) with 0
        self.dataframe["Female_actor_percentage"] = self.dataframe["Female_actor_percentage"].fillna(0)

        # Extract year from Movie_release_date
        self.dataframe["Start_year"] = self.dataframe["Movie_release_date"].astype(str).str.extract(r"(\d{4})").astype(float)

        # Filter movies released after 1970
        filtered_df = self.dataframe[self.dataframe["Start_year"] >1970]
        filtered_df["Movie_genres"] = filtered_df["Movie_genres"].str.split(", ")
        exploded_df = filtered_df.explode("Movie_genres")

        # Exclude "Uncategorized" genres
        exploded_df = exploded_df[exploded_df["Movie_genres"] != "Uncategorized"]

        # Calculate the mean percentage of female actors
        grouped_df = (exploded_df.groupby(["Start_year", "Movie_genres"])["Female_actor_percentage"].mean().reset_index())

        # Bar plot
        fig = px.bar(
            grouped_df,
            x="Start_year",
            y="Female_actor_percentage",
            color="Movie_genres",
            barmode="group",
            title="Evolution of Female Actor Percentage by Genre Over the Years (Post-1970)",
            labels={"Female_actor_percentage": "Percentage of Female Actors (%)", "Start_year" : "Year"},
            height=600,
            width=1000)

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Percentage of Female Actors [%]",
            legend_title="Genre",
            template="plotly_white",
            bargap=0.1,  
            bargroupgap=0.0)

        fig.show()
        #fig.write_html("female_percentage_genre_evolution.html")


    def plotly_kde(self, variables, bins=15, second_dataframe=None, save_html=False, output_dir="./plots"):
        """
        Plot KDE curves for the specified variables using two datasets. 

        Args:
            variables (list): Column names to plot.
            bins (int): Number of bins for the histograms. Default is 15.
            second_dataframe (pd.DataFrame): Second dataframe to compare with self.dataframe
            save_html (bool): Whether to save the plots as HTML files. Default is False.
            output_dir (str): Directory to save the HTML files if save_html is True. Default is "./plots".
        """
       
        if second_dataframe is None:
            raise ValueError("A second dataframe must be provided for comparison.")

        # Create output directory if it does not exist
        if save_html:
            os.makedirs(output_dir, exist_ok=True)

        # Generate individual plots for each variable
        for col in variables:
            if col not in self.dataframe.columns or col not in second_dataframe.columns:
                raise ValueError(f"Column '{col}' is not in one or both DataFrames.")

            # Prepare data
            data_streaming = self.dataframe[col].dropna()
            data_box_office = second_dataframe[col].dropna()

            # KDE Streaming
            kde_streaming = gaussian_kde(data_streaming)
            x_vals_streaming = np.linspace(data_streaming.min(), data_streaming.max(), 500)
            kde_vals_streaming = kde_streaming(x_vals_streaming)

            # KDE Box Office
            kde_box_office = gaussian_kde(data_box_office)
            x_vals_box_office = np.linspace(data_box_office.min(), data_box_office.max(), 500)
            kde_vals_box_office = kde_box_office(x_vals_box_office)

            fig = go.Figure()

            # Add KDE Streaming
            fig.add_trace(
                go.Scatter(
                    x=x_vals_streaming,
                    y=kde_vals_streaming,
                    mode="lines",
                    name="Streaming",
                    line=dict(color="rgba(0, 123, 255, 0.8)", width=2)))

            # KDE Box Office
            fig.add_trace(
                go.Scatter(
                    x=x_vals_box_office,
                    y=kde_vals_box_office,
                    mode="lines",
                    name="Box Office",
                    line=dict(color="rgba(255, 165, 0, 0.8)", width=2)))


            fig.update_layout(
                title_text=f"KDE Curves for {col}",
                xaxis_title=col,
                yaxis_title="Density",
                template="plotly_white",
                showlegend=True,
                legend=dict(itemsizing='constant', traceorder='normal'),
                height=400,
                width=600)


            #if save_html:
            #    output_path = os.path.join(output_dir, f"{col}_kde.html")
            #    fig.write_html(output_path)

            # Display the plot immediately
            fig.show()

    def count_gender_words(self):
        """
        Group data by movie and gender, calculate word counts and percentages.
        """
        #Group by movie and gender
        gender_word_count_df = self.dataframe.groupby(["tconst", "Gender"])["Words"].sum().unstack()
        gender_word_count_df = gender_word_count_df.rename(columns={"f": "Female_word_count", "m": "Male_word_count"})
        
        #Calculate total words per movie
        gender_word_count_df["Total_word_count"] = gender_word_count_df.sum(axis=1)
        
        #Reset index and clean up
        gender_word_count_df = gender_word_count_df.reset_index()
        gender_word_count_df.columns.name = None
        
        #Calculate the percentage of words for each gender
        gender_word_count_df['Word_percentage_men'] = (
            (gender_word_count_df['Male_word_count'] / gender_word_count_df['Total_word_count']) * 100
        )
        gender_word_count_df['Word_percentage_women'] = (
            (gender_word_count_df['Female_word_count'] / gender_word_count_df['Total_word_count']) * 100
        )
        
        self.gender_word_count_df = gender_word_count_df
        return self.gender_word_count_df

    def plotly_gender_words(self):
        gender_word_count_df = self.dataframe
        # Prepare data for plotting
        percentage_df = pd.DataFrame({
            "Percentage": list(gender_word_count_df['Word_percentage_men']) + list(gender_word_count_df['Word_percentage_women']),
            "Gender": ["Men"] * len(gender_word_count_df) + ["Women"] * len(gender_word_count_df)
        })
        # Create the histogram
        fig = px.histogram(
            percentage_df, 
            x="Percentage", 
            color="Gender",
            nbins=50,
            opacity=0.6,
            title="Distribution of Word Percentage Spoken by Men and Women in Movies",
            labels={"Percentage": "Percentage of Words Spoken", "Gender": "Gender"}
        )
        # Update layout for a clean look
        fig.update_layout(
            yaxis_title="Number of Movies",  
            barmode='overlay'  # Overlay bars for transparency
        )
        fig.show()

    def plotly_(self, variables, bins=15, second_dataframe=None, save_html=False, output_dir="./plots"):

            fig.show()
   
