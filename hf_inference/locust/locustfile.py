from locust import HttpUser, task


class ZeroShotEndpoint(HttpUser):
    @task
    def prediction(self):
        self.client.get(
            "/predict",
            json={
                "text": "Just a random text about the president of the USA",
                "categories": ["politics", "sports"]
            }
        )
