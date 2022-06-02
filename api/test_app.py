import unittest

from app import app


class FlaskTest(unittest.TestCase):
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get("/predict")
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)

    def test_index_content(self):
        tester = app.test_client(self)
        response = tester.get("/predict")
        self.assertEqual(response.content_type, "application/json")

    def test_index_data(self):
        tester = app.test_client(self)
        response = tester.get("/predict")
        self.assertTrue(b"Predict Price" in response.data)


if __name__ == "__main__":
    unittest.main()
