
from flask import Flask, escape, request
app = Flask(__name__)

import pandas as pd
import os


df1=pd.read_excel('./공공데이터개방_낱알식별목록_re.xlsx')
print(df1)
    



if __name__ == '__main__':    
    app.run(host='127.0.0.1', port=5000, debug=True)

    
