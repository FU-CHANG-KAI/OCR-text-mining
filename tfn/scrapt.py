import os
import pandas as pd
from bs4 import BeautifulSoup as bs
import numpy as np
import pickle_file 



###### Scrap html to dataframe "books" by using a hierarchical file open paclage: "os.walk"
def scrapt_html_to_df():
    walk_dir = "D:\\Project\\DS\\Data Mining\\cw2\\gap-html\\gap-html\\"
    books = pd.DataFrame(columns=['book_name', 'contents'])
    
    print('walk_dir = ' + walk_dir)
    print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
    
    book_id = 1
    for root, subdirs, files in os.walk(walk_dir):
        path, book_name = os.path.split(root)
        all_text = ""
        for filename in files:
            # os.path.join => join str with //
            file_path = os.path.join(root, filename)
            soup = bs(open(file_path, encoding="utf-8"), "html.parser") #read file
            ocr_tags = soup.select(".ocr_cinfo") #read tags
            text_list = [tag.get_text() for tag in ocr_tags] #extract text
            text = ' '.join(text_list)
            if text.strip() != '':
                all_text = all_text + text + " " #concat text in each page and add to dataframe        
        # print(book_name)
        # write all pages for processed book to dataframe
        books.loc[book_id] = [book_name, all_text]
        print("Save book: "+book_name)
        book_id+=1
    
    return books
   
if __name__ == '__main__':
    df = scrapt_html_to_df()
    pickle_file.save(df, 'books.pkl') 


