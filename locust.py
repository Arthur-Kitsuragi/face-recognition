from locust import HttpUser, task, between

class UploadFilesUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def upload_im(self):
        with open("tests/resources/test.jpg", "rb") as f:
            files = [("files", ("test.jpg", f, "image/jpeg"))]
            data = {"action": "default_action"}
            response = self.client.post("/uploadfiles", files=files, data=data)
            if response.status_code != 200:
                print(f"Error /uploadfiles: {response.status_code}, {response.text}")

class UploadFilesPromptUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def upload_im_with_prompt(self):
        with open("tests/resources/1g.jpg", "rb") as f:
            files = [("files", ("test.jpg", f, "image/jpeg"))]
            data = {"prompt": "Make the man fly"}
            response = self.client.post("/uploadfiles_", files=files, data=data)
            if response.status_code != 200:
                print(f"Error /uploadfiles_: {response.status_code}, {response.text}")
