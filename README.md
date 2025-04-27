# 🚢 Titanic Survival Prediction

Welcome to my **Titanic Survival Prediction** project! I built a machine learning model to predict whether a passenger survived the Titanic disaster using a dataset from Kaggle. This project covers everything from data preprocessing to model evaluation, all documented in a Jupyter Notebook for a hackathon submission. Let’s dive in! 🌊

---

## 📝 Project Overview

The **Titanic Survival Prediction** project is my hackathon submission to predict passenger survival using the `brendan45774/test-file` dataset from Kaggle. I wanted to create a model that handles data cleaning, explores patterns with EDA, trains a Random Forest Classifier, and evaluates it thoroughly. The final model nailed a perfect 100% accuracy, which makes me think the dataset might be a simplified version for learning.

### Objectives 🎯
- Build a classification model to predict survival (0 = didn’t survive, 1 = survived).
- Clean the data by handling missing values, encoding categoricals, and normalizing numericals.
- Dig into the data with EDA to spot trends.
- Check the model’s performance with accuracy, precision, recall, and cross-validation.
- Share it all in a GitHub repo with a detailed notebook (this repo!).

This project is original, built from scratch, and open-source under the MIT License.

---

## 📊 Dataset Details

The dataset (`brendan45774/test-file`) holds Titanic passenger info. Here’s what each column means:

| Column        | Description                              |
|---------------|------------------------------------------|
| `PassengerId` | Unique ID for each passenger.            |
| `Survived`    | Target (0 = didn’t survive, 1 = survived). |
| `Pclass`      | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd), shows socio-economic status. |
| `Name`        | Passenger’s name.                        |
| `Sex`         | Gender (male or female).                 |
| `Age`         | Passenger’s age in years.                |
| `SibSp`       | Number of siblings or spouses aboard.    |
| `Parch`       | Number of parents or children aboard.    |
| `Ticket`      | Ticket number.                           |
| `Fare`        | Ticket fare amount.                      |
| `Cabin`       | Cabin number.                            |
| `Embarked`    | Port (C = Cherbourg, Q = Queenstown, S = Southampton). |

- **Size**: Test set had 84 samples (50 non-survivors, 34 survivors), so the full dataset is likely a few hundred rows.
- **Missing Values**: `Age` (86 missing), `Fare` (1 missing), `Cabin` (327 missing).

---

## 🛠️ Methodology

I followed these steps, all detailed in the Jupyter Notebook (`Titanic_Survival_Prediction.ipynb`).

### 1. **Load the Dataset** 📥
I grabbed the dataset with `kagglehub` and loaded it into pandas.

### 2. **Explore the Data** 🔍
I checked the structure and found missing values in `Age`, `Fare`, and `Cabin`.

### 3. **Preprocess the Data** 🧹
- Filled `Age` and `Fare` with medians.
- Set `Cabin` to 'Unknown' for missing values.
- Encoded `Sex`, `Embarked`, `Pclass` with one-hot encoding.
- Normalized `Age`, `Fare`, `SibSp`, `Parch` with `StandardScaler`.

### 4. **Exploratory Data Analysis (EDA)** 📊
I analyzed the data:
- **Age Distribution**: Slightly right-skewed, most around the median age (likely 28-30 before normalization).
- **Survival by Gender**: Females (~1.0 survival rate) vs. males (~0.2), showing "women and children first."
- **Survival by Class**: Non-2nd class (~0.35) beat 2nd class (~0.30), likely due to 1st class access.
- **Correlation**: `Survived` weakly ties to `Fare` (0.19) and `Parch` (0.16), `Age` barely (0.008).

### 5. **Split the Data** ✂️
I split it 80% training, 20% testing.

### 6. **Train the Model** 🏋️
I picked a Random Forest Classifier (100 estimators). `Sex_male` (0.8689) was the top feature, with `Fare` (0.0494) and `Age` (0.0314) following.

### 7. **Evaluate the Model** 📉
I used accuracy, precision, recall, and a classification report.

### 8. **Optimize and Finalize** ⚙️
I ran 5-fold cross-validation and trained the final model on all data.

---

## 📈 Results

The model did amazingly well, but the perfect scores hint the dataset might be a toy version.

### Performance Metrics
- **Test Set** (84 samples: 50 non-survivors, 34 survivors):
  - Accuracy: 1.00 (100%)
  - Precision: 1.00
  - Recall: 1.00
- **Classification Report**:
  ```
              precision    recall  f1-score   support
  0           1.00      1.00      1.00        50
  1           1.00      1.00      1.00        34
  accuracy                        1.00        84
  ```
- **Cross-Validation**: 5-fold CV scores [1.0, 1.0, 1.0, 1.0, 1.0], mean 1.00, std 0.00.

### Key Insights
- **Gender Rules**: `Sex_male` (0.8689) was the big predictor, matching the "women and children first" policy.
- **Other Factors**: `Fare` and `Age` had small impacts, `Embarked` and `Pclass` barely mattered.
- **Perfect Scores**: 100% accuracy is wild for Titanic data (usually 75-85%). This dataset might be simplified.

---

## 🖥️ How to Run the Project

Here’s how to get it running on your machine.

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Kaggle API

### Steps
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/Code-with-pranav/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Kaggle API**:
   - Install: `pip install kaggle`.
   - Add your API key (see [Kaggle API docs](https://www.kaggle.com/docs/api)).
   - Or download `brendan45774/test-file` manually and add `test-file.csv` here.

### Running
1. **Start Jupyter**:
   ```bash
   jupyter notebook Titanic_Survival_Prediction.ipynb
   ```
2. **Run All Cells**:
   - Open it in your browser (e.g., `http://localhost:8888`).
   - Execute all cells to see code, plots, and results.

### Testing
- Check EDA for age distribution and survival rate plots.
- Look at the evaluation section for metrics (100% accuracy!).
- The final model is trained on all data—ready to use!

### Usage
- Explore the notebook step-by-step.
- Run cells to see the process live.
- Tweak it to try other models or preprocessing.

### Limitations and Future Improvements
- **Dataset**: 100% accuracy suggests it’s simplified. Real data might give 75-85%.
- **Model**: Random Forest might overfit; try logistic regression.
- **EDA**: Add more viz (e.g., survival by family size).
- **Dataset**: Test with the standard Kaggle Titanic dataset.

### Conclusion
My Titanic Survival Prediction project hits all the hackathon marks! It’s got a solid pipeline—preprocessing, EDA, training, and evaluation—with a perfect 100% accuracy. Gender was the star predictor, which fits history. The dataset’s simplicity makes me want to test it on real Titanic data next. Great learning experience!

---

## 📁 Project Structure

```
titanic-survival-prediction/
├── Titanic_Survival_Prediction.ipynb  # Notebook with code and analysis
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

**Note**: The dataset isn’t included. Download it from Kaggle with `kagglehub.dataset_download("brendan45774/test-file")`.

---

## 💡 Observations and Next Steps

The 100% accuracy is a red flag—real Titanic data isn’t this clean. I think this dataset’s a learning tool. Next, I’ll:
- Test on the standard Kaggle Titanic dataset.
- Try a simpler model to confirm results.
- Dig deeper into potential data leakage.

---

## 📜 License

This project is under the MIT License. Check the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Kaggle for the dataset.
- Scikit-learn, pandas, and seaborn for awesome tools.
- The Titanic challenge for being a classic ML problem!
