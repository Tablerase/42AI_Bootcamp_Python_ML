import sys
from io import FileIO


class CsvReader():
    def __init__(self, filename=None, sep=',', header=False, skip_top=0, skip_bottom=0):
        self.filename = filename
        self.sep = sep
        self.is_header: bool = header
        self.skip_top = skip_top
        self.skip_bottom = skip_bottom
        self.file = None
        self.header = []
        self.data = []

    def __enter__(self):
        """Special method when called in 'with' - start of context manager"""
        try:
            self.file: FileIO = open(self.filename, 'r')
            print(f"{'entering file':_^60}")

            lines = self.file.readlines()
            self.header = lines[0].strip().split(self.sep)
            self.header = [item.strip() for item in self.header]
            lines = lines[1:]

            fields = len(self.header)
            for i, line in enumerate(lines):
                record = [field.strip()
                          for field in line.strip().split(self.sep)]
                record = [item
                          for item in record if not len(item) == 0]
                if len(record) != fields:
                    raise ValueError(
                        f"Records amount different from number of fields. line {i + 1}: {record}.fields: {self.header}")
                self.data.append(record)

            # mismatch between numb of records vs numb of fields
            # records with different length
        except (FileNotFoundError, ValueError):
            # raise Exception
            return None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Special method to cleanup - end of context manager"""
        if self.file:
            self.file.close()
        ...

    def getdata(self):
        """ Retrieves the data/records from skip_top to skip bottom.
        Returns:
            nested list (list(list, list, ...)) representing the data.
        """
        if self.skip_top > 0:
            self.data = self.data[self.skip_top:]
        if self.skip_bottom > 0:
            self.data = self.data[:self.skip_bottom]
        return self.data

    def getheader(self):
        """ Retrieves the header from the csv file.
        Returns:
            list: representing the data (when self.header is True).
            None: (when self.header is False).
        """
        if self.is_header == False:
            return None
        else:
            return self.header


if __name__ == "__main__":

    with CsvReader('bad.csv') as file:
        if file == None:
            print("File is corrupted")

    with CsvReader('good.csv', header=True) as file:
        data = file.getdata()
        header = file.getheader()
        print(header)
        for list in data:
            print(list)
