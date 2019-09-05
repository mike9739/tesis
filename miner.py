import twint 
import csv
users =[]
with open('Users.csv') as twUsers:
    ##lee los datos del csv
    reader = csv.DictReader(twUsers)
    for row in reader:
        users.append(row['username'])

# # print(users)    
# with open("users_missing.txt",'r')as file:
#     for line in file:
#         users.append(line.rstrip('\n'))
   
    for user in users:
        c = twint.Config()
        c.Username = user
        c.Since = '2016-01-01'
        c.Store_csv = True
        # Name of the directory
        c.Output = "data"
        

        twint.run.Search(c)





