# with open("./sample.txt", "r") as f:
#     lines = f.readlines()

txt_path = "./sample.txt"

file = open(txt_path, 'r')
readfile = file.read()

id_list = readfile.split("\n")

for id in id_list:
    print(f"Your process with id: {id}")
