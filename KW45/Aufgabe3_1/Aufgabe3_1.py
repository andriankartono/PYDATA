'''
Write a function convert to dtype, which takes a variable number
of arguments and tries to convert them to a specific datatype,
given with a keyword argument dtype. Also there should be a
second keyword-argument debug, which – when set to true – prints
some informative output in case there is a ValueError in the
conversion process. Note that every ValueError should still be
visible in the context calling the function 
'''

def convert_to_dt(*args, dtype, debug=False):
    counter=0
    ls=[]
    isPrint=True
    for i in args:
        if(debug==True):
            try:
                temp=dtype(i)
                ls.append(temp)
                counter+=1
            except ValueError as e:
                txt="ValueError: Cant Convert '{}' at index {}"
                print(txt.format(i, counter))
                raise e
                isPrint=False
        else:
            temp=dtype(i)
            ls.append(temp)

    if(isPrint):
        print("[", end='')
        for a in ls:
            if(a==ls[-1]):
                print(a, end='')
            else:
                #print(type(a))
                print(a, end=', ')
        print(']')
        
if __name__ == "__main__":
    print("Executing as main program")
    i_s = convert_to_dt("1", 4, 5.0, dtype=int)
    print(i_s)
    s_s = convert_to_dt((1,0), "a", 15.1516, dtype=str)
    print(s_s)
    try:
        х = convert_to_dt(5, "a", dtype=int, debug=False)
    except ValueError:
        print("encountered value error")
        pass
    try:
        х = convert_to_dt(5, "a", dtype=int)
    except ValueError:
        print("encountered value error")
        pass


