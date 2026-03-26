# Tutorial Group: RDS2S2 G3
Members:
1. Wong Jin Xuan 2314630
2. Dorcas Lim Yuan Yao 2314535
3. Tan Yen Ping 2314615

# Group 3 - Movie Recommendation System

This prototype includes:
- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation
- Interactive Streamlit UI (run locally)

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Download the following folder from Google Drive (Issue: File too large to be submitted):
   - Pickle/

   Link to Google Drive SharedFolder (RDS2(S2)G3_Group3_Prototype): https://drive.google.com/drive/folders/1eiswTDkSPLNiUH44YWrofNZ8w1dEnXJE?usp=sharing

3. Place the downloaded folder in the same directory as `movie_recommendations.py`

4. Run the app:
   streamlit run movie_recommendations.py

## Notes
- TMDb API key is included in `movie_recommendations.py` for retrieving movie's details.
- Tested with Python 3.10 and Streamlit 1.29.

___________________________________________________________

## Alternative

CleanedData folder with saved preprocessed datasets will be created after running DataPreprocessing.ipynb
- movies_cleaned
- ratings_cleaned

Pickle folder with saved models / model componenets will be created after running Content-Based Filtering.ipynb and Collaborative Filtering.ipynb
- collab_best_model_rmse.pkl
- content_best_config.pkl
- content_movie_indices.pkl
- content_sigmoid_sim_matrix.pkl
- content_tfidf_vectorizer.pkl

Ideally file running sequence:
1. DataPreprocessing.ipynb
2. Content-Based Filtering.ipynb
3. Collaborative Filtering.ipynb
4. Hybrid.ipynb
5. movie_recommendations.py 

___________________________________________________________

Complete Folder Structure (With Large Files) 

Group3/
├── CleanedData/
     ├── movies_cleaned
     ├── ratings_cleaned
├── Pickle/
     ├── collab_best_model_rmse.pkl
     ├── content_best_config.pkl
     ├── content_movie_indices.pkl
     ├── content_sigmoid_sim_matrix.pkl
     ├── content_tfidf_vectorizer.pkl
├── requirements.txt
├── README.txt
├── Collaborative Filtering.ipynb
├── Content-Based Filtering.ipynb
├── DataPreprocessing.ipynb
├── Hybrid.ipynb
├── credits.csv
├── keywords.csv
├── movies_metadata.csv
├── ratings_small.csv
├── movie_recommendations.py


