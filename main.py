#Step 1: Data Preprocessing
import re #regular expression
from collections import Counter
import numpy as np
import pandas as pd

#Implement the function process_data which
#1) Reads in a corpus (text file)  #2) Changes everything to lowercase  3) Returns a list of words.

def process_data(filename):
    words = []
    with open(filename) as f:
        file_name_data = f.read()
    file_name_data = file_name_data.lower()
    words = re.findall('\w+',file_name_data)
    return words    

word_l= process_data('shakespeare.txt')
vocab = set(word_l)
print(f"The first ten words in the text are: \n{word_l[0:10]}")
print(f"There are {len(vocab)} unique words in the vocabulary.")

#a get_count functio that returns a dictionary of word vs frequency
def get_count(word_l):
    word_count_dict = {}
    for word in word_l:
        if word in word_count_dict:
            word_count_dict[word]+=1
        else:
            word_count_dict[word]=1
    return word_count_dict

word_count_dict = get_count(word_l)
print(f"There are {len(word_count_dict)} key values pairs")
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")

#Compute the probability that each word will appear if randomly selected from the corpus of words.
#implement get_probs function

def get_probs(word_count_dict):
    probs={}
    m = sum(word_count_dict.values())
    for key in word_count_dict.keys():
        probs[key] = word_count_dict[key]/m
    return probs    

#Now we implement 4 edit word functions
#delete_letter: given a word, it returns all the possible strings that have one character removed.
#switch_letter: given a word, it returns all the possible strings that have two adjacent letters switched.
#replace_letter: given a word, it returns all the possible strings that have one character replaced by another different letter.
#insert_letter: given a word, it returns all the possible strings that have an additional character inserted.

def delete_letter(word):
    delete_l=[]
    split_l=[]
    for i in range(len(word)):
        split_l.append((word[0:i],word[i:]))
    for a,b in split_l:
        delete_l.append(a+b[1:])
    return delete_l    
delete_word_l = delete_letter(word="cans")

def switch_letter(word):
    split_l=[]
    switch_l=[]
    for i in range(len(word)):
        split_l.append((word[0:i],word[i:]))
    switch_l = [a + b[1] + b[0] + b[2:] for a,b in split_l if len(b) >= 2]
    return switch_l
switch_word_l = switch_letter(word="eta")     

def replace_letter(word):
    split_l=[]
    replace_l=[]
    for i in range(len(word)):
        split_l.append((word[0:i],word[i:]))
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = [a + l + (b[1:] if len(b)> 1 else '') for a,b in split_l if b for l in letters]
    return replace_l

replace_l = replace_letter(word='can')    
        
def insert_letter(word):
    split_l=[]
    insert_l=[]
    for i in range(len(word)+1):
        split_l.append((word[0:i],word[i:]))
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = [a + l + b for a,b in split_l for l in letters]
    #print(split_l)
    return insert_l    
insert_l = insert_letter('at')
print(f"Number of strings output by insert_letter('at') is {len(insert_l)}")

#combining the edits
#switch operation optional
def edit_one_letter(word,allow_switches=True):
    edit_one_set=set()
    edit_one_set.update(delete_letter(word))
    if allow_switches:
        edit_one_set.update(switch_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    return edit_one_set

tmp_word = "at"
tmp_edit_one_set = edit_one_letter(tmp_word)
# turn this into a list to sort it, in order to view it
tmp_edit_one_l = sorted(list(tmp_edit_one_set))

#edit two letters
def edit_two_letters(word, allow_switches=True):
    edit_two_set=set()
    edit_one= edit_one_letter(word, allow_switches=allow_switches)
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w,allow_switches=allow_switches)
            edit_two_set.update(edit_two)
    return edit_two_set
tmp_edit_two_set = edit_two_letters("a")
tmp_edit_two_l = sorted(list(tmp_edit_two_set))
print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
print(f"First 10 strings {tmp_edit_two_l[:10]}")
print(f"Last 10 strings {tmp_edit_two_l[-10:]}") 

#get corrected word
def get_corrections(word,probs,vocab,n=2):
    suggestions=[]
    n_best=[]
    suggestions = list((word in vocab and word) or edit_one_letter(word).intersection(vocab) or edit_two_letters(word).intersection(vocab))
    n_best = [[s,probs[s]] for s in list(reversed(suggestions))]
    return n_best 

my_word = 'dys' 
probs = get_probs(word_count_dict)
tmp_corrections = get_corrections(my_word, probs, vocab, 2)
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}") 

#Now that we have implemented the auto correct functions, lets calculate the similarity between two word
#Minimum edits required to convert one word to another, insert:1, delete:1,replace:2 using dynamic programming

def min_edit_distance(source,target,insert=1,delete=1,replace=2):
    m = len(source)
    n = len(target)
    D = np.zeros((m+1, n+1), dtype=int) 
    for row in range(1,m+1): # Replace None with the proper range
        D[row,0] = D[row-1,0] + delete
    for col in range(1,n+1): # Replace None with the proper range
        D[0,col] = D[0,col-1] + insert
    # Loop through row 1 to row m
    for row in range(1,m+1): 
        
        # Loop through column 1 to column n
        for col in range(1,n+1):
            
            # Intialize r_cost to the 'replace' cost that is passed into this function
            r_cost = replace
            
            # Check to see if source character at the previous row
            # matches the target character at the previous column, 
            if source[row-1] == target[col-1]:
                # Update the replacement cost to 0 if source and target are the same
                r_cost = 0
                
            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            D[row,col] = min([D[row-1,col]+delete, D[row,col-1]+insert, D[row-1,col-1]+r_cost])
          
    # Set the minimum edit distance with the cost found at row m, column n
    med = D[m,n]
    
    
    return D, med
source =  'play'
target = 'stay'
matrix, min_edits = min_edit_distance(source, target)
print("minimum edits: ",min_edits, "\n")
idx = list('#' + source)
cols = list('#' + target)
df = pd.DataFrame(matrix, index=idx, columns= cols)
print(df)    

        
        
        
        


        
        