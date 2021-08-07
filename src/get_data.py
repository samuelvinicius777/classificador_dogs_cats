import os
import pandas as pd
import sqlalchemy

# conectando no banco local
str_connection = 'sqlite:///{path}'

Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_dir = os.path.join(Base_dir, 'data')

print('Diretório do projeto:', Base_dir)
print('Diretório dos dados:', Data_dir)

files_names = [i for i in os.listdir(Data_dir) if i.endswith('.csv')]
print(files_names)

# abrindo conexão com o banco local
str_connection = str_connection.format(path=os.path.join(Data_dir, 'olist.db'))

for i in files_names:
    df_tmp = pd.read_csv(os.path.join(Data_dir, i))
    print(df_tmp.info())
    db_name = 'fr_'+ i.strip('.csv').replace('olist_','').replace('_datasetr','')
    #vai criar o arquivo olist.db na minha lista de arquivos
    df_tmp.to_sql(db_name, str_connection)

