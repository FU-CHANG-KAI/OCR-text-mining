import xlwt
from csv import reader

def list_csv(feature_vectors):
    # Build a csv file and one sheet named 'doc2vec'
    book = xlwt.Workbook()
    sheet = book.add_sheet('doc2vec')

    for i, l in enumerate(feature_vectors):
        for j, col in enumerate(l):
            sheet.write(i, j, col)

    name = 'doc2vec'
    book.save(name)


def csv_list():

    # read csv file as a list of lists
    with open('students.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        print(list_of_rows)