INTRUSION DETECTION SYSTEM – DJANGO + DEEP LEARNING

1. Extract the ZIP file
2. Open terminal in project folder

3. Create virtual environment:
   python -m venv venv

4. Activate venv:
   Windows:
     venv\Scripts\activate

5. Install dependencies:
   pip install -r requirements.txt

6. Run migrations:
   python manage.py makemigrations
   python manage.py migrate

7. Start server:
   python manage.py runserver

8. Open browser:
   http://127.0.0.1:8000/

NOTES:
- Upload 0–4 CSV datasets
- If none uploaded, default datasets are used
- Results are auto-generated per dataset
- Graphs & confusion matrices are stored in /results folder
