##libreria para poder leer csv
import pandas as pd
##diccionarios para guardar los nombes -genero , nombres-nivel educativo
diccionario_niveleducativo ={}
diccionario_genero = {}
##leer el arhivo con los datos de twitter
contador2=0

reader = pd.read_csv('AllUsers.csv')
usernames = reader['username'].values
education = reader['niveleducativo'].values
genders = reader['Sexo'].values

diccionario_genero = dict(zip(usernames,genders))
diccionario_niveleducativo = dict(zip(usernames,education))

##se lee el archivo con los nombres de usuario
with open('names.txt','r') as nombres:
    for linea in nombres.readlines():
        writter=''
        linea = linea.rstrip('\n')
        writter=diccionario_genero[linea]
        writter+='\n'
        with open('gender.txt','a' ) as gender:
                gender.write(writter)       

with open('names.txt','r') as nombres:
    for linea in nombres.readlines():
        writter2=''
        linea = linea.rstrip('\n')
        writter2=diccionario_niveleducativo[linea]
        writter2+='\n'
        with open('nivel_educativo.txt','a' ) as gender:
                    writter=str(writter)
                    gender.write(writter)            

# print(comparador)



    
