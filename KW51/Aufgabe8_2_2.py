'''
send me an email with smtplib from a python-script. You can just
login to an SMTP-Server as shown here. The TUM-server is
postout.lrz .de at port 587. Never store your password in the
code!. (1pts)
'''

import smtplib, ssl

password= input("Type your password and press enter: ")
sender="johanesandrian.kartono@tum.de"
receiver=["felix.mayr@tum.de"]
port = 587
smtp_server= "postout.lrz.de"
message="ge38zum Johanes Andrian Kartono. Task 9.2. Happy New Year"

context= ssl.create_default_context()

with smtplib.SMTP(smtp_server, port)as server:
    server.starttls(context=context)
    server.login(sender, password)
    server.sendmail(sender, receiver, message)