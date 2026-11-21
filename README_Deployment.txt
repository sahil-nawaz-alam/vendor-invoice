Vendor Invoice Intelligence System - Streamlit deployment

1. Arrange files:
   project/
      app.py
      requirements.txt
      models/
         predict_freight_model.pkl
         predict_flag_invoice_model.pkl
         scaler.pkl

2. Install:
   pip install -r requirements.txt

3. Launch:
   streamlit run app.py

Train/export the models using the notebook first. The models directory and files
must match the names above. For Streamlit Community Cloud, push these files to
a GitHub repository, then deploy app.py and set requirements.txt as dependencies.
