#This Code loops from 1 to 100, calculate the square of each even number and the cube of each non-even number

for x in range(1,101):
    if(x%2==1):
        output= x**3
        txt="{}^3 = {}"
        print(txt.format(x,output))
    else:
        output= x**2
        txt="{}^2 = {}"
        print(txt.format(x,output))

