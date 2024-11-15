# "Reel" Realities: How Gender and Age Shape Success Across Box Office and Streaming Platforms
## Abstract

Over the past decade, we have seen a rise in equal gender representation worldwide. This longstanding issue is now being addressed in various ways, including one of the most influential: the film industry. Or at least, so it seems...

Are women truly being represented in an equal way to men in movies? What is the "Reel" reality behind these attempts? Is sexualization truly a thing of the past? How does this vary around the world? We will attempt to tackle these questions in our analysis to get to the bottom of the true intentions of production studios.

Who runs the world? Money. Hence, to answer the questions above, we will look at the impact of gender representation on ROI (Return On Investment) and on our own new "Success" metric. Putting into evidence whether or not a secret formula has been discovered by directors to gain success through gender representation.

## Research questions:
1. How can we measure success? Is success solely based on ROI? On ratings? To answer these questions we will attempt to create our own metric to evaluate success.
2. What is the impact of an actor's gender and age on success, ratings or ROI? How does it vary across time, genre and countries?
3. How does it compare to movies made for streaming platforms? Do they follow a different formula? Have box office movies adapted since the rise of streaming?
4. What are the social reasons behind the presence of female characters in movies? Is it due to sexualization or genuine equality of representation?

## Additional datasets:
<table style="border: 1px solid; border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid; padding: 8px;">Dataset name</th>
    <th style="border: 1px solid; padding: 8px;">URL</th>
    <th style="border: 1px solid; padding: 8px;">Comments</th>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 8px;">Internet Movie Database. (2024). IMDb non-commercial datasets.</td>
    <td style="border: 1px solid; padding: 8px;"><a href="https://developer.imdb.com/non-commercial-datasets/">https://developer.imdb.com/non-commercial-datasets/</a></td>
    <td style="border: 1px solid; padding: 8px;">The IMDB dataset is used to get information that was missing within the CMU dataset. We mainly extracted movie ratings, runtimes, adult ratings and crew information.</td>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 8px;">The Numbers</td>
    <td style="border: 1px solid; padding: 8px;"><a href="https://www.the-numbers.com/movie/budgets">https://www.the-numbers.com/movie/budgets</a></td>
    <td style="border: 1px solid; padding: 8px;">The "The Numbers" dataset gives us budget information about the movies allowing us to estimate the ROI. It is important to note "Budget numbers for movies can be both difficult to find and unreliable." "The data we have is, to the best of our knowledge, accurate but there are gaps and disputed figures." quoted from the website. We were however only able to obtain a free sample of the dataset as we had to pay to get the complete file.</td>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 8px;">Movie Dataset: Budgets, Genres, Insights by Utkarsh Singh</td>
    <td style="border: 1px solid; padding: 8px;"><a href="https://www.kaggle.com/datasets/utkarshx27/movies-dataset/data">https://www.kaggle.com/datasets/utkarshx27/movies-dataset/data</a></td>
    <td style="border: 1px solid; padding: 8px;">This dataset obtained from Kaggle allows us to complete some more missing budget rows.</td>
  </tr>
</table>

## Methods (develop once finished):
1. Regression Analysis: Define success as a binary outcome (revenue to cost ratio), predict the likelihood of a movie's success based on the categories mentioned.
2. Time Series Analysis: Analyze how trends in gender roles, lead actor age, and box office success have changed over time. Could compare to socio-economic events that would have an impact on box office thanks to Google trends.
3. Cluster Analysis (instead of regression analysis): Group movies based on characteristics (genre, gender of lead roles, and character archetypes). How these groupings correlate with success.
4. Text Mining and Natural Language Processing (NLP): Analyze plot summaries and reviews for language indicative of sexualization (e.g., objectifying terms) or agency (e.g., leadership roles).

## Proposed timeline:
- 29/11: Linear regressions completed and analyzed. Success metric revised and finalized
- 06/12: Comparison to streaming platforms.
- 13/12: Sexualization analysis completed, final remarks.
- 20/12: Data story developed, final touches, submission.

## Milestones:
- Sections 2 and 3 (Success metric and Gender/Age impact): Edouard, Serge and Leila
- Section 4 (Comparison with streaming services): Yoan and Serge
- Section 5 (Sexualization study): Fawzia, Leila and Edouard
- Data story: Fawzia and others (TBD)
- Final touches: Everyone

## Questions for TAs:
- What do you think of our success metric? Is it relevant? Do we have any ways of improving it?
- Our main limitaion is the absence of information on revenue and costs, reducing our dataset to a few thousand datapoints. Is that still enough to infer significant results?