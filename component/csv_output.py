from csv import writer

class CSVOutput:
    def __init__(self, path):
        self.file = open(path, 'w', newline='')
        self.writer = writer(self.file)
        column_names = [f'{j}{i}' for i in range(9) for j in 'xyz']
        self.writer.writerow(column_names)

    def process(self, inputs):
        self.writer.writerow(inputs)

    def __del__(self):
        self.file.close()
