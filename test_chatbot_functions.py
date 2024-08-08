import unittest
import pandas as pd
from joblib import load
from fuzzywuzzy import process
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unittest.mock import MagicMock


from app import (
    app, make_token, index_auth, predict_symptom, get_doctor_recommendation, extract_symptom, 
    predict_disease_from_symptom, userSession
)


class TestMakeToken(unittest.TestCase):
    def test_token_length(self):
        token = make_token(16)
        self.assertEqual(len(token), 22)  

    def test_token_uniqueness(self):
        token1 = make_token()
        token2 = make_token()
        self.assertNotEqual(token1, token2)


class TestIndexAuth(unittest.TestCase):
    def test_index_auth(self):
        session_id = index_auth(userSession)
        self.assertIn(session_id, userSession)
        self.assertEqual(userSession[session_id], 0)


class TestPredictSymptom(unittest.TestCase):
    def test_predict_symptom(self):
        symptom_list = ['fever', 'cough', 'headache']
        user_input = 'high fever'
        result = predict_symptom(user_input, symptom_list)
        self.assertEqual(result, 'fever')


class TestGetDoctorRecommendation(unittest.TestCase):
    def test_get_doctor_recommendation(self):
        global df_doctors
        df_doctors = pd.DataFrame({
            'disease': ['Common Cold'],
            'doctor': ['Aaron Odhiambo (Family Medicine)']
        })
        
        disease = 'Common Cold'
        recommendations = get_doctor_recommendation(disease)
        expected = "Recommended doctors: Aaron Odhiambo (Family Medicine)"
        self.assertEqual(recommendations, expected)

    def test_no_recommendation(self):
        disease = 'Unknown Disease'
        recommendations = get_doctor_recommendation(disease)
        expected = "No doctor recommendations available for this disease."
        self.assertEqual(recommendations, expected)


class TestExtractSymptom(unittest.TestCase):
    def test_extract_symptom(self):
        symptom_list = ['fever', 'cough', 'headache']
        user_input = 'head ache'
        result = extract_symptom(user_input, symptom_list)
        self.assertEqual(result, 'headache')


class TestPredictDiseaseFromSymptom(unittest.TestCase):
    def test_predict_disease_from_symptom(self):
        model = MagicMock()
        model.predict.return_value = ['Common Cold']
        
        vectorizer = MagicMock()
        vectorizer.transform.return_value = 'mock_vector'

        symptom_list = ['fever', 'cough']
        with unittest.mock.patch('app.load', side_effect=[model, vectorizer]):
            result = predict_disease_from_symptom(symptom_list)
            self.assertEqual(result, 'Common Cold')


if __name__ == '__main__':
    unittest.main()
