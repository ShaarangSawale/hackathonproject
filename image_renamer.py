import os
def rename(directory):
    os.chdir(directory) # Changing to the directory you specified.
    count=0
    for name in os.listdir(directory):
        print(name)
        
        os.rename(name,"Square"+name)
        
path = "C://Users//antho//hackathonproject//Masterset//Trainset//Square"
rename(path)
