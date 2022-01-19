'''
This Program outputs a hollow triangle as ASCII-ART on the terminal when run.
'''
#improvement possibility: Filter the input
input= int(input("Enter your value: "))
for i in range(1, input+1):
    if(i==input):
        print("*" * (2*input-1), end="")
    elif(i==1):
        print(" " * (input-1), end="")
        print('*', end='')
        print(" " * (input-1))
    else:
        print(' '*(input-i), end='')
        print('*',end='')
        print(' '*(1+2*(i-2)), end='')
        print('*', end='')
        print(" " * (input-1))

