# BDD Climate Control Project

Historically, Ford has had high warranty expenses, more than most other main U.S. auto manufacturers. In 2019 alone, Ford paid out $4.56 billion in warranty claims. The Climate Control team, in particular, is striving to help reduce warranty spending by investigating faulty compressors and other customer claims relating to the AC system. Possible causes for these customer complaints include driving behavior, differences in actual and requested temperature, system blockages, and other extraneous factors (season, duration of trip, etc.).

## Objectives
1. Gain insights into large differences that exist between actual compressor temperature and temperature requested by passenger. 
2. Analyze the duration and frequency of intervals where the compressor temperature is below 0°C.
3. Determine if driver behavior impacts the AC system.

## Workflow
1. Identify vehicle selection criteria.
2. Determine CAN signals of interest.
3. Pull data from Big Data Drive (a GDIA data source containing CAN data transmitted at frequencies of thousands of signals/ms for 3000+ vehicles).
4. Conduct exploratory data analysis to study project objectives.
5. Scale study to other vehicle types.

The documents in this folder contain notebooks used for prelimary work, preprocessing, EDA, and scaling.
