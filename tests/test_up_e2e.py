import unittest
from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class TestSentimentAnalysisAppE2E(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get('http://127.0.0.1:5000/')
        self.driver.implicitly_wait(3)

    def test_home_page(self):
        header_text = self.driver.find_element("tag name", "h1").text
        self.assertIn('Sentiment Analysis App', header_text)

    def test_predict_endpoint(self):
        review_input = self.driver.find_element("name", "review")
        review_input.send_keys('Bad movie!')
        predict_button = self.driver.find_element("css selector", 'input[type="submit"]')
        predict_button.click()
        result_text = self.driver.find_element("tag name", "h2").text
        self.assertIn('The sentiment is Negative', result_text)

    def test_new_comment_redirect(self):
        def test_new_comment_redirect(self):
            try:
                new_comment_button = self.driver.find_element(
                    By.XPATH, "//a[contains(text(), 'New comment')]"
                )
                new_comment_button.click()
                self.assertTrue(self.driver.current_url.endswith('http://127.0.0.1:5000/'))

            except NoSuchElementException as e:
                print(f"Element with link text 'New comment' not found: {e}")
                raise

    def tearDown(self):
        self.driver.quit()


if __name__ == '__main__':
    unittest.main()
