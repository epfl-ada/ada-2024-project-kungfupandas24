# "Reel" Realities: How Gender and Age Shape Success Across Box Office and Streaming Platforms
## Abstract

Over the past decade, we have seen a rise in equal gender representation worldwide. This longstanding issue is now being addressed in various ways, including one of the most influential: the film industry. Or at least, so it seems...

Are women truly being represented in an equal way to men in movies? What is the "Reel" reality behind these attempts? Is sexualization truly a thing of the past? How does this vary around the world? We will attempt to tackle these questions in our analysis to get to the bottom of the true intentions of production studios.

Who runs the world? Money. Hence, to answer the questions above, we will look at the impact of gender representation on ROI (Return On Investment) and on our own new "Success" metric. Putting into evidence whether or not a secret formula has been discovered by directors to gain success through gender representation.

Find the answers to all these questions and more on our [project website](https://edouard-6.github.io)

## Research questions:
1. How can we measure success? Success is often defined by ratings for cinephiles and return on investment (ROI) for investors. Both parties are just as crucial to the true success of a movie with the former being the source of acclaim and the latter the source of funds for production. Why not combine both? Our new WS (Weighted Success) metric attempts to encapsulate the importance of both factors.
2. What are the factors that affect success through each of its definitions? How does it vary from one country to the next? From one production studio to the next? Or even according to the prominent actor genders? Is there a formula for success?
3. How does it compare to movies made for streaming platforms? These movies have less trouble bringing in viewers thanks to their easy and cheap access online. Hence, do they follow a different formula? Have box office movies adapted since the rise of streaming?
4. What are the social reasons behind the presence of female characters in movies? It is common to watch a movie and realize that the female support character may only be present to attract the male gaze. To what extent do production studios abuse of this? The [Bechdel Test](https://www.merriam-webster.com/dictionary/Bechdel%20Test) is a known metric to measure this. However, it is commonly criticised for being too surface level. Is there a better way to measure sexualization in movies? Can we do that using dialogues from movies? What about plot summaries?

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
  <tr>
    <td style="border: 1px solid; padding: 8px;">Polygraph's Film Dialogue Dataset</td>
    <td style="border: 1px solid; padding: 8px;"><a href="https://github.com/matthewfdaniels/scripts/">https://github.com/matthewfdaniels/scripts/</a></td>
    <td style="border: 1px solid; padding: 8px;">This dataset allows us to examine movie dialogues to better understand the representation of women in movies.</td>
  </tr>
    <tr>
    <td style="border: 1px solid; padding: 8px;">Bechdel Dataset</td>
    <td style="border: 1px solid; padding: 8px;"><a href="https://www.kaggle.com/datasets/treelunar/bechdel-test-movies-as-of-feb-28-2023/data">https://www.kaggle.com/datasets/treelunar/bechdel-test-movies-as-of-feb-28-2023/data</a></td>
    <td style="border: 1px solid; padding: 8px;">This dataset obtained from Kaggle allows us to look at the representation and sexualization of women in movies.</td>
  </tr>
</table>

## Methods:
1. Definition of a new metric: The WS (Weighted Success) metric is introduced in [Section 2](results.ipynb#2-our-success-metric) of our analysis. Through financial data we calculate the Return on Investment of each movie. We combine ROI with the ratings obtained from the IMDB dataset.
2. Regression Analysis: We run a regression analysis on both the box office movies ([Section 3](results.ipynb#3-gender-and-age-vs-success)) and movies from streaming platforms ([Section 4](results.ipynb#4-how-does-it-compare-to-streaming-platforms)). The goal is to study the impact of features such as actor gender count, adult rating, movie genre and more on the success of a movie. A different regression model is run using ROI, ratings and Weighted Success to study the varying impact on each metric. We also run it on the Bechdel dataset in Section 5
3. Natural Language Processing (NLP): We analyze plot summaries and reviews for language indicative of sexualization (e.g., objectifying terms) or agency (e.g., leadership roles). We apply zero-shot classification and clustering based on embeddings and TF-IDF matrices of the plot summaries.
   

## Proposed timeline:
- 29/11: Linear regressions completed and analyzed. Success metric revised and finalized
- 06/12: Comparison to streaming platforms.
- 13/12: Sexualization analysis completed, final remarks.
- 20/12: Data story developed, final touches, submission.

## Milestones:
- Sections 2 and 3 (Success metric and Gender/Age impact): Edouard, Serge and Leila
- Section 4 (Comparison with streaming services): Yoan and Serge
- Section 5 (Sexualization study): Fawzia, Leila and Edouard
- Data story: Everyone
- Final touches: Everyone
