import datetime
class ELogger:
    def __init__(self, log_path):
        self.log_path = log_path

    def print(self, msg):
        print(f">>>>>>>>>> {datetime.datetime.now()} : {msg}")
        with open(f"{self.log_path}", 'a') as f:
            f.write(f"{msg}\n")
