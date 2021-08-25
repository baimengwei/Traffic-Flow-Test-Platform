class A:
    cnt = 0

    def __init__(self):
        self.cnt += 1
        print(self.cnt)

    def print_info(self):
        print(self.cnt)
        pass
