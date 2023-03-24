import os
import json
from json import JSONEncoder

intents = []

#------------------------------------------------classes--------------------------#
class IntentsFile:
    def __init__(self, intents):
        self.intents = intents

class Intent:
    def __init__(self, tag, patterns, responses):
        self.tag = tag
        self.responses = responses
        self.patterns = patterns


#subclass JSONEncoder
class IntentsEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__
#---------------------------------------------functions--------------------------#
def ToJSON_File(filename, filecontents):

    if(os.path.exists(filename)):
        os.remove(filename)

    with open(filename,'w') as data:
            data.write(str(filecontents))

def addIntent(tag, corpus, response):
    intents.append(Intent(tag, getPatterns(corpus), [response]))

def createIntents(intentsFilePath):
    myintents = IntentsFile(intents)

    details = json.dumps(myintents, indent=4, cls=IntentsEncoder)
    #make a JSON string with no indentions
    #details = IntentsEncoder().encode(myintents)

    ToJSON_File(intentsFilePath, details)
    print("intents file created...")
    # print(details)


def getPatterns(filename):
    patterns = []
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    for line in Lines:
        patterns.append(line.strip())
    return patterns
