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

def read_abvs(file):
	abvs_d = {}
	with open(file, 'r', encoding = 'utf-8') as file_r:
		for line in file_r:
			token = line.rstrip()
			abvs_d[token.lower()] = 1
	return abvs_d


URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
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
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

work_d = '/home/miguel/Documentos/tesis/'
text_f = work_d+'tweets.txt'
abvs_c_f = work_d+'abreviaturas.txt'
words_f = work_d+'words.txt'
emo_f = work_d+'emoticons.txt'
hash_f = work_d+'hashtags.txt'
at_f = work_d+'ats.txt'
link_f = work_d+'links.txt'
abv_f = work_d+'abvs.txt'

abv_d = read_abvs(abvs_c_f)

i = 0
url_re = re.compile(URLS, re.VERBOSE | re.I | re.UNICODE)
hashtag_re = re.compile('(?:^|\s)[＃#]{1}(\w+)', re.UNICODE)
#mention_re = re.compile('(?:^|\s)[＠@]{1}([^\s#<>[\]|{}]+)', re.UNICODE) # To include more complete names
mention_re = re.compile('(?:^|\s)[＠@]{1}(\w+)', re.UNICODE)
                        
with open(text_f, 'r', encoding = 'utf-8') as text_r, open(words_f, 'w', encoding='utf-8') as words_w, open(emo_f, 'w', encoding='utf-8') as emo_w, open(hash_f, 'w', encoding='utf-8') as hash_w, open(at_f, 'w', encoding='utf-8') as at_w, open(link_f, 'w', encoding='utf-8') as link_w, open(abv_f, 'w', encoding = 'utf-8') as abv_r:
	for line in text_r:
		line = line.rstrip().lower()
		hashs = hashtag_re.findall(line)
		ats = mention_re.findall(line)
		links = url_re.findall(line)
		line = clean(line,hashs,ats,links)
		emoticons = emo_re.findall(line)
		emojis = [w for w in line if w in emoji.UNICODE_EMOJI]
		words = re.findall('[a-záéíóúñàèìòù][a-záéíóúñàèìòù_-]+',line)
		abvs = [w for w in re.findall('[a-z0-9ñáéíóú+/]+',line) if w in abv_d and len(w) > 1]

		words_w.write(' '.join(w for w in words)+'\n')
		abv_r.write(' '.join(w for w in abvs)+'\n')
		emo_w.write(' '.join(w for w in emoticons+emojis)+'\n')
		hash_w.write(' '.join(w for w in hashs)+'\n')
		at_w.write(' '.join(w for w in ats)+'\n')
		link_w.write(' '.join(w for w in links)+'\n')
		i += 1
		print(i)