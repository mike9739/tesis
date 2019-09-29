##libreria para poder leer csv
import pandas as pd
##diccionarios para guardar los nombes -genero , nombres-nivel educativo
diccionario_niveleducativo ={}
diccionario_genero = {}
##leer el arhivo con los datos de twitter



reader = pd.read_csv('AllUsers.csv')
usernames = reader['username'].values
education = reader['niveleducativo'].values
genders = reader['Sexo'].values

diccionario_genero = dict(zip(usernames,genders))
diccionario_niveleducativo = dict(zip(usernames,education))
#reader_names = list(pd.read_csv('Dataset.csv'))
#for linea in reader_names:
#     linea = linea.rstrip('\n')
#     with open('names.txt','w') as names:
#         names.write(linea)


contar=0
##se lee el archivo con los nombres de usuario
with open('names.txt','r') as nombres,open('gender.txt','w' ) as gender:
    for linea in nombres:
        contar = contar+1
        writter=''
        linea = linea.rstrip('\n')
        writter=diccionario_genero[linea]
        writter+='\n'
        gender.write(writter)

with open('names.txt','r') as nombres,open('nivel_educativo.txt','w' ) as nivel:
    for linea in nombres.readlines():
        writter2=''
        linea = linea.rstrip('\n')
        writter2=diccionario_niveleducativo[linea]
        writter2+='\n'
        nivel.write(writter2)
        
       
            

# print(comparador)



    
