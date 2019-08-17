x = open('letter-recognition.data')
s=x.read()
for i in range(1,27):
    s = s.replace(str(chr(64+i)), str(i))
    print(str(i))
x.close()
x=open('letter-recognition.data','w')
x.write(s)
x.close()