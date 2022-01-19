'''
Write a program, which goes through the provided wordlist (as
words.txt - from https://github.com/dwyl/english-words - note
that the github version is new and thus has different
line-numbering) and
• prints all words including “nano” in the order of appearance
(including the line-number (starting from 1!)), like:
@l366980 *nano*: synanons
• (interleaved to that) prints all words starting with “uni” in the order
of appeareance: @l433568 uni*: unintrospectively
• keeps track of the number of “words” including more than 3
hyphens and prints the result out at the end:
More than 3 hyphens: XXXXX (where XXXXX is the actual
number of words)
• find the length of the maximal-length word and print that out at the
end: Max length: ?? (you don't need to keep track of the maximal
length word)
'''


file1 = open("D:/python/KW43/aufgabe2_3/Output2_3.txt","w")
mf = open("D:/python/KW43/aufgabe2_3/words.txt", 'r')
file_line = mf.readline()
line_number=1
max_length=0
counter=0

while(file_line):
    #check if nano is in a word
    length=len(file_line.strip())
    if("nano" in file_line):
        output="@l{} *nano*: {}"
        file1.write(output.format(line_number, file_line))

    #check for words starting with uni
    if(file_line.startswith("uni")):
        output="@l{} *uni*: {}"
        file1.write(output.format(line_number, file_line))
    
    #check if there is a longer word than the current one
    if(length>max_length):
        max_length=length

    #check for the number of hyphen in a word and if it is more than 3 increment the counter
    hyphen_counter=0
    for i in range(0,length):
        if(file_line[i]=="-"):
            hyphen_counter=hyphen_counter+1
    if(hyphen_counter>3):
        counter=counter+1

    line_number=line_number+1
    file_line=mf.readline()

hyphen="the maximum length of the word is {}\n"
file1.write(hyphen.format(max_length))    

txt2="there are {} words with 3 or more hyphen"       
file1.write(txt2.format(counter))

file1.close()
mf.close()
