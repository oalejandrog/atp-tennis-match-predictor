# Final Project Report: ATP Tennis Match Prediction using Deep Learning

## 1. Main Objective
The main objective of this analysis is to develop a Deep Learning model capable of predicting the winner of a professional ATP tennis match based on pre-match statistics. By leveraging historical data, player rankings, and physical attributes, we aim to provide a predictive tool that can benefit stakeholders such as:
- **Sports Analysts**: To identify key factors contributing to match outcomes.
- **Tournament Organizers**: To estimate competitive balance in draws.
- **Matchmaking Platforms (e.g., MatchPe)**: To improve player pairing and competitive equity.

This project focuses on a **Deep Neural Network (Multi-Layer Perceptron)** approach to handle the non-linear relationships between player attributes and match success.

## 2. Dataset Description
The analysis uses the **Jeff Sackmann ATP Tennis Dataset**, specifically professional match results from **2018 to 2024**.
- **Attributes**: The dataset includes features such as player rankings, ranking points, height, age, hand (left/right), court surface (Hard, Clay, Grass), and tournament level (Grand Slam, Masters, etc.).
- **Goal**: Predict a binary outcome (1 if "Player A" wins, 0 if "Player B" wins) using a balanced dataset where the order of players is randomized to avoid bias.
- **Size**: Approximately 18,877 match records were loaded, resulting in a balanced training set after preprocessing.

## 3. Data Exploration and Preprocessing
### Data Exploration
- **Rankings**: There is a clear correlation between the rank difference and the likelihood of winning. Higher-ranked players (lower numerical rank) win significantly more often.
- **Physical Attributes**: Player height and age show moderate influence, with certain surfaces favoring different physical profiles.

### Preprocessing Actions
- **Filtering**: Selected matches from 2018-2024 to ensure relevance to modern tennis.
- **Cleaning**: Addressed missing values in height by using the median and filled missing hand information with a placeholder ('U').
- **Feature Engineering**: 
    - Created a balanced dataset by randomly assigning "Player A" and "Player B" roles.
    - Encoded categorical variables (Surface, Hand, Tournament Level) using Label Encoding.
    - Scaled numerical features (Rank, Points, Age, Height) using `StandardScaler` to ensure efficient Neural Network training.
- **Splitting**: Data was split into Training (70%), Validation (15%), and Test (15%) sets.

## 4. Deep Learning Model Variations
Three variations of the Multi-Layer Perceptron (MLP) architecture were trained for 30 epochs:

| Model Variation | Architecture | Key Features | Test Accuracy |
|-----------------|--------------|--------------|---------------|
| **Model 1 (Shallow)** | [32, 16] Neurons | Simple, low complexity | 62.82% |
| **Model 2 (Dropout)** | [64, 32, 16] Neurons | Deeper, includes Dropout (0.2) for regularization | **63.36%** |
| **Model 3 (BatchNormalization)** | [128, 64, 32] Neurons | Wider, includes Batch Normalization | 60.89% |

## 5. Final Model Recommendation
I recommend **Model 2 (Dropout)** as the final model. 
- **Performance**: It achieved the highest test accuracy (63.36%).
- **Generalization**: The inclusion of Dropout layers helped prevent overfitting compared to the wider BatchNormalization model, which showed signs of instability in the validation loss curves.
- **Efficiency**: It maintains a good balance between model depth and computational cost.

## 6. Key Findings and Insights
- **Ranking Dominance**: Player rank remains the most significant predictor. However, a Deep Learning model captures nuances that simple rank-based heuristics might miss.
- **Complexity vs. Performance**: Increasing model width (Model 3) did not translate to better performance, suggesting that the current feature set is the bottleneck rather than model capacity.
- **Surface Sensitivity**: Categorical encoding of surfaces allowed the model to adjust weightings for attributes like height, which can be more advantageous on specific surfaces (e.g., Grass).

## 7. Suggestions for Next Steps
- **Feature Engineering (Elo Ratings)**: Incorporating Elo ratings or dynamic "recent form" metrics (e.g., win rate in the last 10 matches) would likely significantly boost accuracy above 70%.
- **Temporal Analysis**: Using a Recurrent Neural Network (RNN) or LSTM could capture the temporal nature of a player's career trajectory and seasonal performance.
- **Head-to-Head (H2H)**: Adding historical H2H records between specific players would provide critical context for specific matchups.
- **Hyperparameter Tuning**: Implementing a Keras Tuner or Grid Search to find the optimal learning rate and dropout percentage.
