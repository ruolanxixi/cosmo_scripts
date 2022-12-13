import ftplib
import numpy as np

server = ftplib.FTP_TLS('arthurhou.pps.eosdis.nasa.gov','ruolan.xiang@env.ethz.ch','ruolan.xiang@env.ethz.ch')
# print("CONNECTED TO FTP")

fsv = 'https://arthurhouhttps.pps.eosdis.nasa.gov'

f = open('filename.txt', 'w')
for i in {'2001', '2002', '2003', '2004', '2005'}:
    for j in {'01', '03', '05', '07', '08', '10', '12'}:
        for k in {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                  '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'}:
            files = server.nlst(f'/gpmdata/{i}/{j}/{k}/imerg/')
            for ii in range(len(files)):
                print(f'{fsv}' + files[ii])

for i in {'2001', '2002', '2003', '2004', '2005'}:
    for j in {'04', '06', '09', '11'}:
        for k in {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                  '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'}:
            files = server.nlst(f'/gpmdata/{i}/{j}/{k}/imerg/')
            for ii in range(len(files)):
                print(f'{fsv}' + files[ii])

for i in {'2001', '2002', '2003', '2005'}:
    for j in {'02'}:
        for k in {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                  '21', '22', '23', '24', '25', '26', '27', '28'}:
            files = server.nlst(f'/gpmdata/{i}/{j}/{k}/imerg/')
            for ii in range(len(files)):
                print(f'{fsv}' + files[ii])

for i in {'2004'}:
    for j in {'02'}:
        for k in {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                  '21', '22', '23', '24', '25', '26', '27', '28', '29'}:
            files = server.nlst(f'/gpmdata/{i}/{j}/{k}/imerg/')
            for ii in range(len(files)):
                print(f'{fsv}' + files[ii])
