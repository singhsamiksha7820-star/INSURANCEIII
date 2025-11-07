# Insurance Policy Status Streamlit Dashboard

This Streamlit app lets you:

1. Explore **5 different charts** to understand insurance portfolio risk and performance.
2. Compare **Decision Tree, Random Forest and Gradient Boosted Trees** on policy status prediction.
3. Upload a **new dataset** and get predicted `POLICY_STATUS` with download support.

## Files

- `app.py` – main Streamlit application.
- `Insurance_excel.xlsx` – sample dataset used by the app.
- `requirements.txt` – Python dependencies (no version pins).

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal.

## Deploying on Streamlit Cloud

1. Push these files to a GitHub repository (no folders required).
2. Create a new Streamlit app from that repo.
3. Set the main file to `app.py`.
4. Deploy – the dashboard should load using `Insurance_excel.xlsx` in the root folder.
