import unittest
from flask import Flask
from flask_app import flask_app  # Изменено: используйте имя объекта Flask-приложения, а не 'app'

class TestSentimentAnalysisApp(unittest.TestCase):

    def setUp(self):
        # Создайте тестовый клиент для приложения
        self.app = flask_app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        print(f"Response status code: {response.status_code}")
        print(f"Response data: {response.data}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sentiment Analysis App', response.data)

    def test_predict_endpoint(self):
        response = self.app.post('/predict', data={'review': 'Great movie!'})
        self.assertEqual(response.status_code, 200)
        # Проверьте, что в ответе есть текст "The sentiment is Positive"
        self.assertIn(b'The sentiment is Positive', response.data)

    def test_new_comment_redirect(self):
        response = self.app.get('/new_comment')
        self.assertEqual(response.status_code, 302)
        # Проверьте, что редирект ведет на ожидаемый URL
        self.assertTrue(response.location.endswith('http://127.0.0.1:5000/'))

if __name__ == '__main__':
    unittest.main()
