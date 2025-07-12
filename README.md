# Most Important Variables (MVP ver_1.09)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

🎯 **Most Important Variables** to aplikacja Streamlit do szybkiego przeglądu danych, wykrywania typu problemu (klasyfikacja/regresja), budowy najlepszego modelu ML oraz podglądu najważniejszych cech (feature importance).  
Obecna wersja to **MVP** – podstawowa funkcjonalność bez zaawansowanych opcji i optymalizacji UI.

---

## 🚀 Funkcje
- ✅ Upload plików `.csv` i `.data`
- ✅ Automatyczne wykrywanie separatora i nagłówków kolumn
- ✅ Wybór kolumny docelowej (target)
- ✅ Rozpoznanie typu problemu: klasyfikacja lub regresja
- ✅ Automatyczne trenowanie najlepszego modelu ML (PyCaret)
- ✅ Wyświetlenie: 
  - Feature Importance (wykres)
  - Leaderboard modeli
  - Przykładowych predykcji
- 🔜 Placeholder dla opisu wyników (wersja v2.x będzie generować AI summary)

---

## 🖥️ Wymagania
- Python **>=3.9,<3.12**
- Pakiety:
  - `streamlit`
  - `pandas`
  - `pycaret`
  - `pillow`
  - (opcjonalnie) `python-dotenv` dla obsługi API key w przyszłości

---

## 📦 Instalacja

1. Sklonuj repozytorium:
    ```bash
    git clone https://github.com/Hiu1975/miv-app.git
    cd miv-app
    ```

2. Stwórz i aktywuj środowisko (rekomendowane):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. Zainstaluj zależności:
    ```bash
    pip install -r requirements.txt
    ```

4. Uruchom aplikację:
    ```bash
    streamlit run app_miv.py
    ```

---

## 📁 Struktura projektu

