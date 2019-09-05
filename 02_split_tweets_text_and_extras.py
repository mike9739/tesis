# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:39:56 2018

@author: JC

@description: split a line of text in words, ats, hashtags, links
                and emoticons, creating a file for each element.
"""
import re
from nltk.tokenize.casual import EMOTICON_RE as emo_re
import emoji
import codecs 
import chardet

def clean(line,hashs,ats,links):
    for link in links:
        line = line.replace(link,'')
    for has_h in hashs:
        line = line.replace('#'+has_h,'')
        line = line.replace('＃'+has_h,'')
    for at in ats:
        line = line.replace('@'+at,'')
        line = line.replace('＠'+at,'')
    return line

URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[\-][a-z0-9]{4,})
    [.]
    (?:[/][a-z]{7,})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""


"""
ruta='Enron'
dataSet='enron'#+month

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta
text_file = main_dir+'/full_data.txt'
words_file = main_dir+'/'+dataSet+'_words.txt'
emo_file = main_dir+'/'+dataSet+'_emoticons.txt'
hash_file = main_dir+'/'+dataSet+'_hashtags.txt'
at_file = main_dir+'/'+dataSet+'_ats.txt'
link_file = main_dir+'/'+dataSet+'_links.txt'
"""

dataSet='DataSetTest'

main_dir='/home/miguel/Documentos/thesis/'
text_file = main_dir+'/tweets.txt'
words_file = main_dir+'/'+dataSet+'_words.txt'
emo_file = main_dir+'/'+dataSet+'_emoticons.txt'
hash_file = main_dir+'/'+dataSet+'_hashtags.txt'
at_file = main_dir+'/'+dataSet+'_ats.txt'
link_file = main_dir+'/'+dataSet+'_links.txt'




i = 0
url_re = re.compile(URLS, re.VERBOSE | re.I | re.UNICODE)
hashtag_re = re.compile('(?:^|\s)+[＃#]{1}(\w+)', re.UNICODE)
#mention_re = re.compile('(?:^|\s)[＠@]{1}([^\s#<>[\]|{}]+)', re.UNICODE) # To include more complete names
mention_re = re.compile('(?:^|\s)+[＠@]{1}(\w+)', re.UNICODE)
                        
with open(text_file,'r',encoding='utf-8') as text_reader, open(words_file,'w', encoding='utf-8') as words_writer, open(emo_file, 'w',encoding='utf-8') as emo_writer, open(hash_file,'w',encoding='utf-8') as hash_writer, open(at_file,'w',encoding='utf-8') as at_writer, open(link_file,'w',encoding='utf-8') as link_writer:
    for line in text_reader:
        line = line.rstrip().lower()
        hashs = hashtag_re.findall(line)
        ats = mention_re.findall(line)
        links = url_re.findall(line)
        line = clean(line,hashs,ats,links)
        emoticons = emo_re.findall(line)
        emojis = [w for w in line if w in emoji.UNICODE_EMOJI]
        words = re.findall('[a-záéíóúñàèìòù][a-záéíóúñàèìòù_-]+',line) #Revisar para remover ats, hashs y links
        
        words_writer.write(' '.join(w for w in words)+'\n')
        emo_writer.write(' '.join(w for w in emoticons+emojis)+'\n')
        hash_writer.write(' '.join(w for w in hashs)+'\n')
        at_writer.write(' '.join(w for w in ats)+'\n')
        link_writer.write(' '.join(w for w in links)+'\n')
   
        i += 1
        print(i)