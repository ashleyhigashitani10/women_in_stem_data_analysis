# Women in STEM Data Analysis
## Project Overview
This project analyzes women in stem data; More specifically global representation in 4 major STEM Fields (Biology, Computer Science, Engineering & Math) across countries (China, Canada, the US, Germany, India, and Australia).
The goal of this project was to examine Enrollment & Graduation trends and explore potential socioeconomic influences.

## Into the Data
This analysis involved:
- Data cleaning
- Data quality checks before analysis
- Analysis by grouping & correlation
- Data visualization
- Dataset merging (women_in_stem with GDP dataset)

Overall, data cleaning and quality analysis was mandatory before starting analysis and creating merges.

In this Final_Project branch it was ensured that all scripts were ran from cell 1 for accurate output and flow.

Women_in_stem dataset: 
https://www.kaggle.com/datasets/bismasajjad/womens-representation-in-global-stem-

GDP dataset:
https://data.worldbank.org/indicator/NY.GDP.PCAP.CD

## Key Insights
- There was no linear relatonship between female graduation & enrollment with gender gap index or GDP per capita
- Female enrollment was similar across all income groups (found during quartile-based analysis)
- No strong linear pattern between graduation and enrollment
- Enrollment is similar across fields (Mathematics with the highest enrollment rate, Biology with the lowest)

## Decisions
The main branch contains the base analysis of the women_in_stem dataset. This script went more in depth on checking data quality (because this is the primary dataset) and analysis into the data. File_Structure also contains the primary analysis script but also the GDP merge; The GDP script had a focus on cleaning the GDP data structure and merging it accordingly with women_in_stem. Analysis was made but limited in depth in comparison to the main. This is why I've decided not to merge these 2 branches, both have stronger focuses in 2 different aspects of this analysis. In other words, Main branch had a strong focus on analytical clarity, while File-Structure has a more technical approach to this project. This final branch consolidates the 2 clean scripts and is organized in a folder structure.

## Future Improvements
- Diving deeper in trends in year
- Further analysis in GDP
- Explore other variables/influences 
