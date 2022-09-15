ofile=open("Story.txt","r")
k=ofile.readlines()
t=reversed(k)
for i in t:
   print(i.rstrip())