# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 14:46:25 2016

@author: Yuan Liang
"""

model_tst = 0
model_begin = 108


'''validation data postprocessing'''

new_data_file = open('case2_postprocess.tsv', 'w')
new_data_file.write('user_id\tarticle_id\trevision_id\tnamespace\ttimestamp\n') # header

#   data manipulation 
from datetime import datetime
raw_data_file = open('validation.tsv')
raw_data_file.readline()
active_editors = set()
for line in raw_data_file:
    attr = line.strip().split('\t')
    user = int(attr[0])
    article = int(attr[1])
    revision = int(attr[2])
    namespace = int(attr[3])
    timestamp_dt = datetime.strptime(attr[4], '%Y-%m-%d %H:%M:%S')   
    dt = (2008-timestamp_dt.year)*12 + 1-timestamp_dt.month +(2-timestamp_dt.day)/30.0
    attr[4] = dt
#    timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
    timestamp_str = float(attr[4])    
    new_data_file.write('%d\t%d\t%d\t%d\t%.2f\n' % (user, article, revision, namespace, timestamp_str))
new_data_file.close();

datafile = open('case2_postprocess.tsv')
datafile.readline()
for line in datafile:
    attr = line.strip().split('\t')
    user = int(attr[0])
    if user in active_editors:
        continue 
    timestamp = float(attr[4])
    if (timestamp >= model_tst) and (timestamp <= model_begin):     #TODO
        active_editors.add(user)
datafile.close()



#   initialize

#existing_article = set()
users = sorted(active_editors)  # pool of active users
#from collections import defaultdict





user_edits = {} #edit history
user_edit = {}  #simplified history
user_solus = {} #edit count
user_frstedit = {}  
user_lastedit = {}
user_durating = {}  #first-last
user_recentedit_times = {}  #recent times
user_recent_editdata = {}   #weighted edit data
user_averagedate = {}   #average data
user_standardd = {}
user_article = {}
user_target = {}    #number of editing in lastest 5 month

for user in users:
    user_edits[user] = []
    user_edit[user] = []
    user_solus[user] = 0
    user_recentedit_times[user] = 0
    user_recent_editdata[user] = 0
    user_averagedate[user] = 0
    user_standardd[user] =0
    user_article[user] = 0
    user_durating[user] = 0
    user_frstedit[user] = model_tst
    user_lastedit[user] = model_begin
    user_target[user] = 0



'''validation_result data postprocessing'''

result_data = open('validation_solutions.csv')
result_data.readline()
for line in result_data:
    attr = line.strip().split(',')
    user = int(attr[0])
    solution = int(attr[1])
    if user not in active_editors:
        continue
    user_target[user] = solution
result_data.close()
    



#   feature construction
data_file = open('case2_postprocess.tsv')
data_file.readline()  # header
for line in data_file:
    attr = line.strip().split('\t')
    user = int(attr[0])
    article = int(attr[1])
    timestamp = float(attr[4])


# edit times
    user_solus[user] += 1

# edit recent_editdata
    user_recent_editdata[user] = 1/timestamp + user_recent_editdata[user]
        
# recent edit times
    if (timestamp >= model_tst) and (timestamp <= model_tst+5):
        user_recentedit_times[user] = user_recentedit_times[user] + 1

# average data
    user_averagedate[user] = user_averagedate[user] + timestamp

# edit history and simplified history
    m = float(timestamp)
    user_edits[user].append(m)
    user_edit[user].append(article)

# age
    if timestamp > user_frstedit[user]:
        user_frstedit[user] = timestamp
    if timestamp < user_lastedit[user]:
        user_lastedit[user] = timestamp

data_file.close()


import math
for user in users:
       sum_up = 0
       user_durating[user] = user_frstedit[user] - user_lastedit[user]
       if user_durating[user] < 0:
           user_durating[user] = model_begin - model_tst
       if user_recent_editdata[user] != 0:
           user_recent_editdata[user] = 1/(user_recent_editdata[user]/user_solus[user])
       else:
           user_recent_editdata[user] = model_begin
       if user_averagedate[user] != 0:
           user_averagedate[user] = user_averagedate[user]/(user_solus[user])
       else: 
           user_averagedate[user] = model_begin
       
       if user_solus[user] != 0:
           sum_up = sum([(t-user_averagedate[user])**2 for t in user_edits[user]])
           user_standardd[user] = math.sqrt(sum_up/(user_solus[user]))
       else:
           user_standardd[user] = 0
           
new_data = open('case2_features.tsv', 'w')
new_data.write('user_id\ttrain_target\tedit_duration\trecent_edit_times\tweighted_edit_times\taverage_data\tstrd_edit_derivation\n') # header
for user in users:
    new_data.write('%d\t%d\t%.2f\t%d\t%.2f\t%.2f\t%.2f\n' % (user, user_target[user], user_durating[user], user_recentedit_times[user], user_recent_editdata[user], user_averagedate[user], user_standardd[user]))

new_data.close()

