# SMS Spam Detection

A machine learning project to detect spam messages in SMS texts.

## Project Structure

```
sms_spam_detection/
│
├── data/                      # Contains the dataset
│   ├── spam.csv              # The SMS spam dataset
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory_data_analysis.ipynb  # EDA notebook
│   ├── model_training.ipynb  # Model training notebook
│
├── src/                      # Source code
│   ├── data_preprocessing.py # Data preprocessing functions
│   ├── feature_extraction.py # Feature extraction functions
│   ├── model.py             # Model training and evaluation
│   ├── utils.py             # Utility functions
│
├── requirements.txt          # Python dependencies
└── main.py                  # Main script
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```







## Usage

1. Place your dataset in the `data` directory
2. Run the main script:
```bash
python main.py
```


<br><br>




![image](https://github.com/user-attachments/assets/bb4fb498-efb3-4cff-a733-36f68af69b9b)<br><br>
![image](https://github.com/user-attachments/assets/a4291dbe-a107-4f9b-abe8-0af115cab1d0)
